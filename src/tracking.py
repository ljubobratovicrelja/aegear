import sys
import pickle

import cv2
import numpy as np
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def ProcessFrameForMotionEstimation(frame):
    #f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = cv2.bilateralFilter(frame, -1, 9.0, 5.0)
    return f  # cv2.medianBlur(, 5)


trackingPoint = None


def SelectTrackingPoint(frame):
    global trackingPoint

    winTitle = "Select tracking point"
    cv2.namedWindow(winTitle)

    def onMouse(event, x, y, flags, param):
        global trackingPoint
        if event == cv2.EVENT_LBUTTONDOWN:
            trackingPoint = (x, y)

    cv2.setMouseCallback(winTitle, onMouse)

    while True:
        cv2.imshow(winTitle, frame)

        if trackingPoint is not None:
            drawPoint = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.circle(drawPoint, trackingPoint, 9, [255, 0, 0], 2, cv2.LINE_AA)
            cv2.imshow(winTitle, drawPoint)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyWindow(winTitle)


IMG_HEIGHT = 32
IMG_WIDTH = 32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

calibrationDataPath = "data/calibration.xml"
dataPath = "data/videos/2016_0718_200947_002"  # input video path
sampleVideoPath = "data/videos/K9.MOV"
startFrame = 3000
motionThreshold = 25
maxDistance = 100
output_video = "data/videos/tracking_example"
model_path = "data/model_cnn3.pth"

# load pytorch model
model = torch.load(model_path)
model.eval()

# read calibration data
storage = cv2.FileStorage(calibrationDataPath, cv2.FILE_STORAGE_READ)
mtx = storage.getNode("mtx").mat()
dist_params = storage.getNode("dist").mat()
storage.release()

trackingParams = dict(
    winSize=(9, 9),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

videoStream = cv2.VideoCapture(sampleVideoPath)

assert videoStream.isOpened(), "Failed opening video at {}".format(sampleVideoPath)

ret, prevFrame = videoStream.read()
assert ret, "Failed reading first frame."

imrows, imcols = prevFrame.shape[0:2]

prevFrame = ProcessFrameForMotionEstimation(prevFrame)
trackingPoint = None

# HOG descriptor and support vector classifier
sampleWindow = 32
hog = cv2.HOGDescriptor((sampleWindow, sampleWindow), (16, 16), (8, 8), (16, 16), 8)

with open("data/svc.pickle", "rb") as f:
    svc = pickle.load(f)


def inBounds(cx, cy, s2):
    global imrows
    global imcols

    p1 = (cy - s2, cx - s2)
    p2 = (cy + s2, cx + s2)

    if p1[0] < 0 or p1[1] < 0:
        return False
    if p2[0] >= imrows or p2[1] >= imcols:
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


def drawTrajectory(frame, trajectory, thickness=1):
    if len(trajectory) < 9:
        return frame

    trajectory = np.array(trajectory)
    x = trajectory[:, 0]
    y = trajectory[:, 1]

    x = savgol_filter(x, 9, 3)
    y = savgol_filter(y, 9, 3)

    x = list(map(lambda v: int(v), x))
    y = list(map(lambda v: int(v), y))

    trajectory = list(zip(x, y))

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


# skipping frames towards the start frame
endReading = False
while startFrame > 0:
    ret, frame = videoStream.read()
    startFrame -= 1

    if not ret:
        endReading = True
        break

if endReading:
    print("Skipping way too many frames")
    sys.exit()

skipFrames = 5
trajectory = []

WIN_NAME = "Trajectory Tracking"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

# select frame and calculate pixel to cm ratio
frame = cv2.undistort(frame, mtx, dist_params)
frame = ProcessFrameForMotionEstimation(frame)

pts = []


def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        print("Point selected: {}".format((x, y)))


cv2.setMouseCallback(WIN_NAME, select_points)

while True:
    draw_frame = np.copy(frame)

    for pt in pts:
        cv2.circle(draw_frame, pt, 5, (0, 255, 0))

    cv2.imshow(WIN_NAME, draw_frame)
    if cv2.waitKey(1) == ord("q") or len(pts) == 4:
        break

if len(pts) != 4:
    print("You must select 4 points")
    sys.exit()

persp_T = None
sample_pts = np.array(pts, dtype=np.float32)

real_pts = np.array(
    [[0, 0], [149.0, 5.0], [149.0, 35.0], [0.0, 40.0]], dtype=np.float32
)

img_scaling_factor = (sample_pts[1, 0] - sample_pts[0, 0]) / (
    real_pts[1, 0] - real_pts[0, 0]
)

# move points to match starting x position of samples, and scale up to image scale
transformed_real_pts = real_pts * img_scaling_factor + sample_pts[0, :]

print("Sample points: {}".format(sample_pts))
print("Real points: {}".format(transformed_real_pts))

# do perspective transform to rectify image
persp_T = cv2.getPerspectiveTransform(sample_pts, transformed_real_pts)
image_transformed = cv2.warpPerspective(frame, persp_T, frame.shape[0:2][::-1])

# also warp points to be able to calculate pixel to cm ratio
# add homogeneous coordinate
sample_pts = np.hstack((sample_pts, np.ones((4, 1))))
sample_pts = np.dot(persp_T, sample_pts.T).T

# divide by homogeneous coordinate
sample_pts = sample_pts[:, 0:2] / sample_pts[:, 2].reshape((4, 1))
print("Transformed sample points: {}".format(sample_pts))

# draw transformed samples on transformed image to prove all is transformed ok
for i in range(4):
    pt = (int(sample_pts[i, 0]), int(sample_pts[i, 1]))
    cv2.circle(image_transformed, pt, 5, (255, 255, 0))

cv2.imshow("Transformed", image_transformed)
cv2.waitKey(0)
cv2.destroyWindow("Transformed")

# now calculate pixel to cm ratio
pixel_to_cm_ratio = np.linalg.norm(real_pts[1, :] - real_pts[0, :]) / np.linalg.norm(
    sample_pts[1, :] - sample_pts[0, :]
)

print("Pixel to cm ratio: {}".format(pixel_to_cm_ratio))

# Define the codec and create a VideoWriter object
frame_height, frame_width = frame.shape[0:2]

# Detect the OS
if sys.platform.startswith('win'):
    # Windows
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video + ".avi", fourcc, 24.0, (frame_width, frame_height))
elif sys.platform.startswith('darwin'):
    # macOS
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video + ".mp4", fourcc, 24.0, (frame_width, frame_height))
else:
    # Linux or other
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video + ".avi", fourcc, 24.0, (frame_width, frame_height))

travelDistance = 0.0  # cm
while True:
    readFrame = 0

    while True:
        ret, frame = videoStream.read()
        if not ret:
            endReading = True
            break
        if readFrame == skipFrames:
            break

        readFrame += 1

    if endReading:
        break

    # undistort frame
    frame = cv2.undistort(frame, mtx, dist_params)
    frame = ProcessFrameForMotionEstimation(frame)
    frame = cv2.warpPerspective(frame, persp_T, frame.shape[0:2][::-1])

    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gprevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)

    # calculate distance
    dist = (np.abs(gframe.astype(np.float32) - gprevFrame.astype(np.float32))).astype(
        np.uint8
    )

    _, dst = cv2.threshold(dist, motionThreshold, 255, cv2.THRESH_BINARY)

    # morphological closing
    dst = cv2.erode(dst, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    dst = cv2.dilate(dst, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)))

    dst = cv2.GaussianBlur(dst, (19, 19), 5.0)
    _, dst = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)

    # track further if there is a tracking point

    # find contours
    contours, _ = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    positives = []
    dframe = frame.copy()

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

            if output > 0.7:
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
                if abs(cx1 - frame_width / 2) > abs(cx2 - frame_width / 2):
                    cx, cy = this_frame_positives[1][0]
                else:
                    cx, cy = this_frame_positives[0][0]
            else:
                cx, cy = this_frame_positives[0][0]

            color = (0, 255, 0)
            positives.append((cx, cy))

    dframe = cv2.drawContours(dframe, goodContours, -1, (0, 255, 0), 1, cv2.LINE_AA)
    dframe = cv2.drawContours(dframe, badContours, -1, (0, 0, 255), 1, cv2.LINE_AA)

    # sample prev frame
    prevFrame = frame

    if positives:
        if trackingPoint is None:
            if len(positives) == 1:
                trackingPoint = positives[0]
        else:
            distances = map(
                lambda p: np.linalg.norm(np.array(p) - np.array(trackingPoint)),
                positives,
            )
            positives = sorted(zip(positives, distances), key=lambda p: p[1])

            if positives[0][1] < maxDistance:
                trackingPoint = positives[0][0]

        if trackingPoint:
            trajectory.append(trackingPoint)

    dframe = drawTrajectory(dframe, trajectory)
    travelDistance = trajectoryLength(trajectory) * pixel_to_cm_ratio

    cv2.putText(
        dframe,
        "Travel Distance (cm): {:.2f}".format(travelDistance),
        (30, imrows - 50),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (190, 250, 150),
        1,
        cv2.LINE_AA,
    )

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
