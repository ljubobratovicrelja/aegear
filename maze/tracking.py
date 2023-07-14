import sys
sys.path.append(".")

import os

import cv2
import numpy as np
from scipy.signal import savgol_filter

import torch
from torchvision import transforms

from maze.blockmatching import BlockMatching
from maze.classifier import Classifier
from maze.mazecalibration import MazeCalibration


TRACKING_POINT = None
IMROWS = 0
IMCOLS = 0
WIN_NAME = "Trajectory Tracking"

def ProcessFrameForMotionEstimation(frame):
    return cv2.bilateralFilter(frame, -1, 9.0, 5.0) 


def SelectTrackingPoint(frame):
    global TRACKING_POINT

    winTitle = "Select tracking point"
    cv2.namedWindow(winTitle)

    def onMouse(event, x, y, flags, param):
        global TRACKING_POINT
        if event == cv2.EVENT_LBUTTONDOWN:
            TRACKING_POINT = (x, y)

    cv2.setMouseCallback(winTitle, onMouse)

    while True:
        cv2.imshow(winTitle, frame)

        if TRACKING_POINT is not None:
            drawPoint = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.circle(drawPoint, TRACKING_POINT, 9, [255, 0, 0], 2, cv2.LINE_AA)
            cv2.imshow(winTitle, drawPoint)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyWindow(winTitle)


def inBounds(cx, cy, s2):
    global IMROWS
    global IMCOLS

    p1 = (cy - s2, cx - s2)
    p2 = (cy + s2, cx + s2)

    if p1[0] < 0 or p1[1] < 0:
        return False
    if p2[0] >= IMROWS or p2[1] >= IMCOLS:
        return False

    return True


def trajectoryLength(trajectory):
    if len(trajectory) < 2:
        return 0.0

    prev = trajectory[0]
    sumlength = 0.0
    for p in trajectory[1:]:
        sumlength += np.linalg.norm(np.array(p) - np.array(prev))
        prev = p

    return sumlength


def smoothTrajectory(trajectory, filterSize=15):
    if len(trajectory) < filterSize:
        return trajectory

    trajectory = np.array(trajectory)

    x = trajectory[:, 0]
    y = trajectory[:, 1]

    x = savgol_filter(x, filterSize, 3)
    y = savgol_filter(y, filterSize, 3)

    x = list(map(lambda v: int(v), x))
    y = list(map(lambda v: int(v), y))

    trajectory = list(zip(x, y))

    return trajectory

def drawTrajectory(frame, trajectory, thickness=1):
    if len(trajectory) < 2:
        return frame

    dframe = np.copy(frame)
    prev = trajectory[0]

    intensityIncrement = 1.0 / len(trajectory)

    for i, p in enumerate(trajectory[1:]):
        # calculate color
        h = 29 + int(150.0 * (i * intensityIncrement))
        s = 255
        v = 255
        bgr = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
        color = bgr[0, 0]
        color = (int(color[0]), int(color[1]), int(color[2]))

        cv2.line(dframe, prev, p, color, thickness, cv2.LINE_AA)
        prev = p

    return dframe


calibrationDataPath = "data/calibration.xml"
dataPath = "data/videos/2016_0718_200947_002"  # input video path
sampleVideoPath = "data/videos"
videosToTrack = ["K9.MOV"] # "S1.MOV", "EE1.MOV", "EE3.MOV", "EE4.MOV"]
startFrame = 3000
motionThreshold = 10
maxDistance = 100
inlierThreshold = 0.85
trajectorySmoothingFactory = 11
output_video = "data/videos/tracking_{}"
model_path = "data/model_cnn4.pth"
outputFps = 30.0
detectionFrames = 2  # every n-th frame we track
skipFrames = 1
sampleWindow = 32

# load pytorch model
model = torch.load(model_path)
model.to("cpu")

model.eval()

# create block matching object
blockMatching = BlockMatching(0.5, 16, 16)

# create maze characterization object
maze_calibration = MazeCalibration(calibrationDataPath)

endReading = False

frame = None
prevFrame = None

cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

for videoName in videosToTrack:
    videoPath = os.path.join(sampleVideoPath, videoName)
    outputVideoPath = os.path.join(output_video.format(videoName))

    videoStream = cv2.VideoCapture(videoPath)
    assert videoStream.isOpened(), "Failed opening video at {}".format(videoPath)

    trajectory = []
    pts = []

    ret, frame = videoStream.read()
    if not ret:
        # end of video
        break

    # select frame and calculate pixel to cm ratio
    frame = ProcessFrameForMotionEstimation(frame)

    # reset image size and tracking point set to 
    IMROWS, IMCOLS = frame.shape[0:2]

    pixel_to_cm_ratio = maze_calibration.calibrate(frame)
    print("Pixel to cm ratio: {}".format(pixel_to_cm_ratio))

    frame = maze_calibration.rectify_image(frame)

    # first assign as previous frame, then read next frame
    prevFrame = frame.copy()

    cv2.imshow("Transformed", frame)
    cv2.waitKey(0)
    cv2.destroyWindow("Transformed")

    # Detect the OS
    if sys.platform.startswith('win'):
        # Windows
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(outputVideoPath + ".avi", fourcc, outputFps, (IMCOLS, IMROWS))
    elif sys.platform.startswith('darwin'):
        # macOS
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(outputVideoPath + ".mp4", fourcc, outputFps, (IMCOLS, IMROWS))
    else:
        # Linux or other
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(outputVideoPath + ".avi", fourcc, outputFps, (IMCOLS, IMROWS))

    travelDistance = 0.0  # cm
    frameId = 0
    hasTracking = False  # if tracking is lost, NN inference is performed

    while True:
        readFrame = 0

        while True:
            ret, frame = videoStream.read()
            frameId += 1
            if not ret:
                endReading = True
                break
            if readFrame == skipFrames:
                break

            readFrame += 1

        if endReading:
            break

        # undistort frame
        frame = ProcessFrameForMotionEstimation(frame)
        frame = maze_calibration.rectify_image(frame)

        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gprevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)

        # calculate motion
        motion = blockMatching.detect(gprevFrame, gframe)
        motion = cv2.normalize(motion, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("motion", motion)
        cv2.waitKey()

        # drawing frame
        dframe = frame.copy()

        if frameId % detectionFrames == 0:
            # calculate distance
            dist = (np.abs(gframe.astype(np.float32) - gprevFrame.astype(np.float32))).astype(
                np.uint8
            )

            _, dst = cv2.threshold(dist, motionThreshold, 255, cv2.THRESH_BINARY)

            # morphological closing
            dst = cv2.erode(dst, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            dst = cv2.dilate(dst, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))

            dst = cv2.GaussianBlur(dst, (19, 19), 5.0)
            _, dst = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)

            # track further if there is a tracking point

            # find contours
            contours, _ = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            positives = []

            if contours:

                this_frame_positives = []
                min_area = 800
                max_area = 3000

                goodContours = []
                badContours = []

                for c in contours:
                    area = cv2.contourArea(c)

                    if area < min_area or area > max_area:
                        badContours.append(c)
                        continue
                
                    goodContours.append(c)

                    x, y, w, h = cv2.boundingRect(c)
                    cx = int(x + w / 2)
                    cy = int(y + h / 2)
                    s2 = int(sampleWindow / 2)

                    if not inBounds(cx, cy, s2):
                        continue

                    bb = (cx - s2, cy - s2, sampleWindow, sampleWindow)
                    roi = frame[
                        cy - s2 : cy - s2 + sampleWindow, cx - s2 : cx - s2 + sampleWindow, :
                    ]
                    
                    # Define the transformations
                    transform = transforms.Compose([
                        transforms.ToPILImage(),  # Convert np array to PIL Image
                        transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
                    ])

                    # Apply transformations to the image
                    image_transformed = transform(roi)
                    image_transformed = image_transformed.unsqueeze(0)  # Add batch dimension

                    # Pass the transformed image to the model
                    output = model(image_transformed)

                    if output > inlierThreshold:
                        this_frame_positives.append(((cx, cy), output))
                
                if len(this_frame_positives) != 0:
                    # sort by output
                    this_frame_positives = sorted(this_frame_positives, key=lambda p: p[1], reverse=True)

                    # if there's two, take the one that's closer to the center of the image
                    # this is to avoid mirror image issues
                    cx = None
                    cy = None
                    if len(this_frame_positives) == 2:
                        cx1, cy1 = this_frame_positives[0][0]
                        cx2, cy2 = this_frame_positives[1][0]
                        if abs(cx1 - IMCOLS / 2) > abs(cx2 - IMROWS / 2):
                            cx, cy = this_frame_positives[1][0]
                        else:
                            cx, cy = this_frame_positives[0][0]
                    else:
                        cx, cy = this_frame_positives[0][0]

                    color = (0, 255, 0)
                    positives.append((cx, cy))
                    dframe = cv2.circle(dframe, (int(cx), int(cy)), 5, (255, 255, 0), -1)

                if positives:
                    TRACKING_POINT = positives[0]
                    trajectory.append(TRACKING_POINT)
                    hasTracking = True

                dframe = cv2.drawContours(dframe, goodContours, -1, (0, 255, 0), 1, cv2.LINE_AA)
                dframe = cv2.drawContours(dframe, badContours, -1, (0, 0, 255), 1, cv2.LINE_AA)


        s_trajectory = smoothTrajectory(trajectory, trajectorySmoothingFactory)
        dframe = drawTrajectory(dframe, s_trajectory)
        travelDistance = trajectoryLength(s_trajectory) * pixel_to_cm_ratio

        cv2.putText(
            dframe,
            "Travel Distance (cm): {:.2f}".format(travelDistance),
            (30, IMROWS - 50),
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            (190, 250, 150),
            1,
            cv2.LINE_AA,
        )

        # sample prev frame
        prevFrame = frame

        out.write(dframe)

        cv2.imshow(WIN_NAME, dframe)
        cv2.imshow("movement", dst)

        ret = cv2.waitKey(1)
        if ret == ord("q"):
            break

    print("Final travel distance: {:.2f} cm".format(travelDistance))

    out.release()
    videoStream.release()

cv2.destroyAllWindows()
