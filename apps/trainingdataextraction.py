"""
This script is used to extract training data from videos. It is used to extract
training data for the motion detection module. The script allows the user to
select a point on the video frame and extract a sample of the video frame
centered at that point. The user can also select a region of the video frame
and extract a sample of the video frame centered at the center of the selected
region. The script also allows the user to jump to the next video file in the
data collection descriptor and the previous video file in the data collection
descriptor. The script also allows the user to quit the script.

Instructions:

1. Run the script with the data collection descriptor as the first argument.
    The data collection descriptor is a JSON file that contains the following
    fields:

    a. videosRoot: The root directory of the video files.
    b. videoFiles: A list of video files to extract training data from.
    c. skipFrames: The number of frames to skip when extracting training data.
    d. trainingDataRoot: The root directory of the training data.
    e. sampleWindow: The size of the sample window to extract from the video
        frame.
    
    Example data collection descriptor:

    {
        "videosRoot": "C:/Users/JohnDoe/Desktop/videos/",
        "videoFiles": ["video1", "video2", "video3"],
        "skipFrames": 10,
        "trainingDataRoot": "C:/Users/JohnDoe/Desktop/trainingdata/",
        "sampleWindow": 64
    }

2. The script will load the first video file in the data collection descriptor
    and display the first frame of the video file. The script will also display
    the number of inlier samples and the number of outlier samples. The script
    will also display the name of the video file being processed.

3. The user can select a point on the video frame and extract a sample of the
    video frame centered at that point by left clicking on the video frame,
    to select an inlier (ie. fish sample), or right click wherever on the video
    frame to select an outlier sample (ie the background). The user can also
    jump to the next video file in the data collection descriptor by pressing
    the 'n' key. The user can also jump to the previous video file in the data
    collection descriptor by pressing the 'p' key. The user can also quit the
    script by pressing the 'q' key.

4. The script will save the extracted sample to the training data root
    directory. The script will also update the number of inlier samples and the
    number of outlier samples. The script will also update the training package
    file. The script will also display the name of the video file being
    processed.

5. Repeat steps 3 and 4 until all video files in the data collection descriptor
    have been processed.
"""

import os
import sys

# needed to load maze modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json

import cv2

from moviepy.editor import VideoFileClip

import maze.motiondetection as md


import json

# load data collection descriptor
if len(sys.argv) < 2:
    print("Usage: python trainingdataextraction.py <data collection descriptor>.json")
    sys.exit()

assert os.path.exists(sys.argv[1]), "Data collection descriptor {} does not exist".format(sys.argv[1])

data_collection_descriptor = None
with open(sys.argv[1], 'r') as f:
    data_collection_descriptor = json.load(f)

assert data_collection_descriptor is not None, "Data collection descriptor {} is empty".format(sys.argv[1])

####################################################################################################
# scripts setup

# unpack the descriptor
videosRoot = data_collection_descriptor["videosRoot"]
videoFiles = data_collection_descriptor["videoFiles"]
skipFrames = data_collection_descriptor["skipFrames"]
trainingDataRoot = data_collection_descriptor["trainingDataRoot"]
sampleWindow = data_collection_descriptor["sampleWindow"]
sampleWindow = (sampleWindow, sampleWindow)

# constants
INLIER_NAME = "fish"
OUTLIER_NAME = "background"
WIN_TITLE = "Training Frame Selection"

# variables
frame = None
trackingPoint = None
jumpFrame = False
videoFile = None

good_contours = []
bad_contours = [] 

videoIndex = 0  # index of the video file from the descriptor

training_package = {"background": {}, "fish": {}}

# if training package file exists, load it
if os.path.exists("{}/training.json".format(trainingDataRoot)):
    with open("{}/training.json".format(trainingDataRoot), "r") as f:
        training_package = json.load(f)

# motion detection module
motion_detector = md.MotionDetector(10, 3, 15, 400, 3000)

numInliers = len(list(os.listdir(os.path.join(trainingDataRoot, INLIER_NAME))))
numOutliers = len(list(os.listdir(os.path.join(trainingDataRoot, OUTLIER_NAME))))

print("Inlier samples: {}, Outlier Samples: {}".format(numInliers, numOutliers))


def onMouse(event, x, y, flags, param):
    global trackingPoint
    global frame
    global numInliers
    global numOutliers
    global jumpFrame
    global training_package
    global videoFile

    global good_contours
    global bad_contours

    # store point for drawing
    trackingPoint = None

    sx2 = int(sampleWindow[0]/2)
    sy2 = int(sampleWindow[1]/2)

    trackingPoint = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        contours_combined = good_contours + bad_contours

        if contours_combined:  # if any contours out there
            # find center of the contour
            for contour in contours_combined:
                # if x, y in contour, take that contour's center
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    M = cv2.moments(contour)
                    x = int(M['m10']/M['m00'])
                    y = int(M['m01']/M['m00'])
                    trackingPoint = (x, y)
                    break

        sample = frame[y-sy2:y+sy2+1, x-sx2:x+sx2+1]

        imgPath = os.path.join(trainingDataRoot, INLIER_NAME, "sample_{:06d}.png".format(numInliers))
        cv2.imwrite(imgPath, sample)
        numInliers = numInliers + 1
        training_package["fish"][videoFile].append((x, y))
        jumpFrame = True

    elif event == cv2.EVENT_RBUTTONDOWN:
        sample = frame[y-sy2:y+sy2+1, x-sx2:x+sx2+1]

        imgPath = os.path.join(trainingDataRoot, OUTLIER_NAME, "sample_{:06d}.png".format(numOutliers))
        cv2.imwrite(imgPath, sample)
        numOutliers = numOutliers + 1
        training_package["background"][videoFile].append((x, y))
    else:
        return

    print("Inlier samples: {}, Outlier Samples: {}".format(numInliers, numOutliers))


while True:
    videoIndex = max(0, min(videoIndex, len(videoFiles)-1))
    videoFile = videoFiles[videoIndex]

    videoPath = os.path.join(videosRoot, videoFile)

    if os.path.exists(videoPath) is False:
        print("Video file {} does not exist".format(videoPath))
        videoIndex = videoIndex + 1
        continue

    # initiate the dictionaries
    if videoFile not in training_package["background"]:
        training_package["background"][videoFile] = []
    if videoFile not in training_package["fish"]:
        training_package["fish"][videoFile] = []

    video = VideoFileClip(videoPath)

    goToNextVideo = False
    goToPrevVideo = False
    quitReading = False
        
    numImages = int(video.duration * video.fps)

    def update(frame_id):
        global frame
        global good_contours
        global bad_contours

        if frame_id == 0:
            frame = video.get_frame(0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(WIN_TITLE, frame)
            return

        prevFrame = video.get_frame((frame_id - 1) / video.fps)
        frame = video.get_frame(frame_id / video.fps)
        nextFrame = video.get_frame((frame_id + 1) / video.fps)

        # to BGR
        prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_RGB2BGR)
        frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)
        nextFrame = cv2.cvtColor(nextFrame, cv2.COLOR_RGB2BGR)

        # calculate motion
        good_contours, bad_contours = motion_detector.detect2(prevFrame, frame, nextFrame)

        # draw contours
        frame_draw = frame.copy()
        cv2.drawContours(frame_draw , good_contours, 0, [0, 255, 0], 2)
        cv2.drawContours(frame_draw , bad_contours, 0, [0, 0, 255], 2)

        # draw centers of all contours as crosses
        for contour in good_contours:
            M = cv2.moments(contour)
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            frame_draw = cv2.drawMarker(frame_draw, (x, y), [255, 255, 255], cv2.MARKER_CROSS, 10, 2)

        cv2.imshow(WIN_TITLE,frame_draw )

    cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_TITLE, onMouse)
    cv2.createTrackbar('frame', WIN_TITLE, 0, numImages-1, update)

    update(0)

    while True:
        ret = cv2.waitKey(1) & 0xFF

        if ret == ord('q'):
            quitReading = True
            break
        if ret == ord('n'):
            goToNextVideo = True
            break
        if ret == ord('p'):
            goToPrevVideo = True
            break

        # if array left is pressed fo to next frame
        if ret == ord('a'):
            frameIdx = cv2.getTrackbarPos('frame', WIN_TITLE)
            frameIdx = max(frameIdx-skipFrames, 0)
            cv2.setTrackbarPos('frame', WIN_TITLE, frameIdx)
            update(frameIdx)

        # if arrow right is pressed go to next frame
        if ret == ord('d'):
            frameIdx = cv2.getTrackbarPos('frame', WIN_TITLE)
            frameIdx = min(frameIdx+skipFrames, numImages-1)
            cv2.setTrackbarPos('frame', WIN_TITLE, frameIdx)
            update(frameIdx)

        if jumpFrame:
            jumpFrame = False
            frameIdx = cv2.getTrackbarPos('frame', WIN_TITLE)
            frameIdx = min(frameIdx+1, numImages-1)
            cv2.setTrackbarPos('frame', WIN_TITLE, frameIdx)
            update(frameIdx)

    cv2.destroyWindow(WIN_TITLE)

    if goToNextVideo:
        videoIndex = min(videoIndex+1, len(videoFiles)-1)
        continue

    if goToPrevVideo:
        videoIndex = max(videoIndex-1, 0)
        continue

    if quitReading:
        break


cv2.destroyAllWindows()

# write the training package to a file
with open("{}/training.json".format(trainingDataRoot), "w") as f:
    json.dump(training_package, f, indent=4)
