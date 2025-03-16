"""
Camera calibration script.

This script is used to calibrate the camera and save the calibration parameters
to an opencv file. The calibration parameters are used in the main script to
undistort the images.
 
The first input box defines input parameters for all of the procedure. Be sure to
set the correct path to the calibration video, the number of chessboard corners
for the pattern it was used, and the length of the chessboard squares in cm.

Descriptor file is a json file that contains the following fields:
    - videoPath: path to the video file
    - cornerDataPath: path to the corner data file
    - calibrationDataPath: path to the calibration data file
    - patternSize: number of corners in the calibration pattern (rows, columns)
    - subPixWinSize: depends on the resolution, 5, 5 seems ok for HD camera resolution
    - patternSquareSideLength: length of pattern square side in cm
    - skipFrames: how many frames do we skip while seeking video for calibration shots

Make sure to populate the descriptor file with the correct values according to your
data capture setup.
"""

import sys
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from moviepy.editor import VideoFileClip

import json


assert len(sys.argv) == 2, "Usage: python {} <path_to_descriptor>".format(sys.argv[0])

descriptor = sys.argv[1]

with open(descriptor, "r") as f:
    descriptor = json.load(f)

videoPath = descriptor["videoPath"]  # input video path
cornerDataPath = descriptor["cornerDataPath"] if "cornerDataPath" in descriptor else None  # output path for the corner data, in case we want to redo only calibration without having to detect corners
calibrationDataPath = descriptor["calibrationDataPath"]  # output path for the calibration data
readExistingCornerData = cornerDataPath is not None and os.path.exists(cornerDataPath) # if True, read corner data from cornerDataPath, otherwise detect corners from video

patternSize = descriptor["patternSize"]  # number of corners in the calibration pattern (rows, columns)
subPixWinSize = descriptor["subPixWinSize"]  # depends on the resolution, 5, 5 seems ok for HD camera resolution

patternSquareSideLength = 1.6  # length of pattern square side in cm
skipFrames = 120  # how many frames do we skip while seeking video for calibration shots
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


"""
Open the video stream, and fetch needed metadata for futher processing.
"""

video = VideoFileClip(videoPath)

numFrames = int(video.fps * video.duration)

print("Video has {} frames. Sampling every {}th frame, meaning we will process {} frames.".format(numFrames, skipFrames, numFrames // skipFrames))

"""
Reading video frames, and detecting the chessboard pattern in them.
"""

validFrames = []
allCorners = []  # collection of all corners for calibration purposes

if not readExistingCornerData:

    # Enable interactive mode
    for frame_id in tqdm(range(0, numFrames, skipFrames)):
        frame = video.get_frame(float(frame_id) / video.fps)

        # convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        quitReading = False

        # try detecting the chart
        patternWasFound, corners = cv2.findChessboardCorners(frame, patternSize, cv2.CALIB_CB_FAST_CHECK)

        if patternWasFound:
            corners = cv2.cornerSubPix(frame, corners, subPixWinSize, (-1, -1), criteria)
            allCorners.append(corners)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(frame_bgr, patternSize, corners, patternWasFound)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            validFrames.append(frame)

    allCorners = np.array(allCorners)
    if cornerDataPath is not None:
        try:
            np.save(cornerDataPath, allCorners)
            print("Saved corners to {}".format(cornerDataPath))
        except:
            print("Failed to save corners to {}".format(cornerDataPath))
    else:
        print("Not saving corners to disk, since no path was provided.")

else:
    allCorners = np.load(cornerDataPath)

numSamples = allCorners.shape[0]
print("Collected {} corner sampels.".format(numSamples))

"""
Setting up the object points we need for calibration.
"""
objPoints = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
objPoints[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2).astype(np.float32)
objPoints *= patternSquareSideLength

# Assuming `objPoints` is a list of lists or a list of arrays
# First, convert it to a numpy array
objPoints = np.array([objPoints] * numSamples)

objPoints = np.reshape(objPoints, (numSamples, objPoints[0].shape[0], 1, 3))

# Then, convert it to a 3-channel float32 array if it's not already
if objPoints.dtype != np.float32 or objPoints.shape[-1] != 3:
    objPoints = objPoints.astype(np.float32)


"""
Finally, we run the calibration.
"""

print("Running calibration...")

frameSize = validFrames[0].shape[::-1]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, allCorners, frameSize, None, None)

if ret:
    print("Calibration successful.")
    print("Matrix:\n{}".format(mtx))
    print("Distortion:\n{}".format(dist))

    storage = cv2.FileStorage(calibrationDataPath, cv2.FILE_STORAGE_WRITE)
    storage.write("mtx", mtx)
    storage.write("dist", dist)
    storage.release()

    print("Saved calibration data to {}".format(calibrationDataPath))
else:
    assert False, "Calibration failed"


"""
Finally let's undistor an image sample and display it side by side with the original, to make sure our distorition coefficients are correctly estimated.
"""

# Assuming you have an image stored in the variable `img`
original_img = validFrames[0].copy()

# Compute the undistortion and rectification transformation map
undistorted_img = cv2.undistort(original_img, mtx, dist)

# show original and undistorted image side by side, with labels for every subplot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(original_img, cmap="gray")
ax[1].imshow(undistorted_img, cmap="gray")

ax[0].set_title("Original")
ax[1].set_title("Undistorted")

plt.show()


