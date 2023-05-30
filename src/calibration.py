import sys
import math
import numpy as np
import cv2

####################################################################################################
# scripts setup

videoPath = "data/videos/2016_0718_200947_002.MOV"  # input video path
cornerDataPath = "data/corners.npy"
calibrationDataPath = "data/calibration.xml"
readExistingCornerData = True

patternSize = (8, 6)  # number of corners in the calibration pattern (rows, columns)
subPixWinSize = (5, 5)  # depends on the resolution, 5, 5 seems ok for this camera resolution
maxNumSamples = 40  # if there are more than this collected samples, all samples are shuffled and subsampled
skipFrames = 30  # how many frames do we skip while seeking video for calibration shots
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# setting up object points
patternSquareSideLength = 1.6
objPoints = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
objPoints[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2).astype(np.float32)
objPoints *= patternSquareSideLength

####################################################################################################
# read calibration sequence


vidstream = cv2.VideoCapture(videoPath)

assert vidstream.isOpened(), "Failed to open the video stream at {}".format(videoPath)

# read sample video for out frame size
ret, frame = vidstream.read()
frameSize = frame.shape[0:2]

allCorners = []  # collection of all corners for calibration purposes

if not readExistingCornerData:
    while True:

        quitReading = False
        for skipFrame in range(skipFrames):
            ret, frame = vidstream.read()
            if not ret:
                quitReading = True
                break

        if quitReading:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # try detecting the chart
        patternWasFound, corners = cv2.findChessboardCorners(gray, patternSize, cv2.CALIB_CB_FAST_CHECK)

        if patternWasFound:
            corners = cv2.cornerSubPix(gray, corners, subPixWinSize, (-1, -1), criteria)
            allCorners.append(corners)
            cv2.drawChessboardCorners(frame, patternSize, corners, patternWasFound)

        cv2.imshow("video", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    numSamples = len(allCorners)

    print("Collected {} corner sampels.".format(numSamples))

    allCorners = np.array(allCorners)
    np.save(cornerDataPath, allCorners)

    print("Saved corners to {}".format(cornerDataPath))

    # clean windows
    cv2.destroyAllWindows()
else:
    allCorners = np.load(cornerDataPath)
    numSamples = allCorners.shape[0]

vidstream.release()

# take just number of samples
if numSamples > maxNumSamples:
    #np.random.shuffle(sampleIds)

    ratio = int(math.ceil(numSamples / maxNumSamples))
    print("ratio: {}".format(ratio))

    sampleIds = np.arange(0, numSamples, ratio)

    allCorners = allCorners[sampleIds]
    numSamples = allCorners.shape[0]

    print("Num samples trimmed to {}, according to selection:\n{}".format(numSamples, sampleIds))

# now let's try to calibrate
objPoints = [objPoints] * numSamples
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, allCorners, frameSize, None, None)

if ret:
    print("Matrix:\n{}".format(mtx))
    print("Distortion:\n{}".format(dist))

    storage = cv2.FileStorage(calibrationDataPath, cv2.FILE_STORAGE_WRITE)
    storage.write("mtx", mtx)
    storage.write("dist", dist)
    storage.release()

    #print(rvecs)
    #print(tvecs)
else:
    print("Calibration failed")
    sys.exit(1)

# find optimal camera matrix
h, w = frame.shape[0:2]
#newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort a frame sample
undistoredFrame = cv2.undistort(frame, mtx, dist, None, None)

meanError = 0.0
for i in range(numSamples):
    imgpoints2, _ = cv2.projectPoints(objPoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(allCorners[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    meanError += error
print("Reprojection total error: {}".format(meanError/numSamples))

cv2.imshow("raw (distorted)", frame)
cv2.imshow("undistorted", undistoredFrame)
cv2.waitKey()
