import sys
import pickle

import cv2
import numpy as np
from scipy.signal import savgol_filter
import sklearn.svm as svm


def ProcessFrameForMotionEstimation(frame):
    f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = cv2.bilateralFilter(f, -1, 9.0, 5.0)
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


calibrationDataPath = "data/calibration.xml"
dataPath = "data/videos/2016_0718_200947_002"  # input video path
sampleVideoPath = "data/videos/EE3.MOV"
startFrame = 3000
motionThreshold = 10
maxDistance = 100

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

    # calculate distance
    dist = (np.abs(frame.astype(np.float32) - prevFrame.astype(np.float32))).astype(
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
    dframe = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if contours:
        dframe = cv2.drawContours(dframe, contours, -1, (255, 0, 255), 1, cv2.LINE_AA)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            s2 = int(sampleWindow / 2)

            if not inBounds(cx, cy, s2):
                continue

            bb = (cx - s2, cy - s2, sampleWindow, sampleWindow)
            roi = frame[
                cy - s2 : cy - s2 + sampleWindow, cx - s2 : cx - s2 + sampleWindow
            ]
            desc = hog.compute(roi)
            res = svc.predict([desc])

            if res and res[0] == 1:
                color = (0, 255, 0)
                positives.append((cx, cy))
            else:
                color = (0, 0, 255)

            # cv2.rectangle(dframe, bb, color, 1, cv2.LINE_AA)

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
    travelDistance = trajectoryLength(trajectory)

    cv2.putText(
        dframe,
        "Travel Distance (px): {:.2f}".format(travelDistance),
        (30, imrows - 50),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (190, 250, 150),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow(WIN_NAME, dframe)
    # cv2.imshow("movement", dst)

    ret = cv2.waitKey(1)
    if ret == ord("q"):
        break

videoStream.release()
cv2.destroyAllWindows()
