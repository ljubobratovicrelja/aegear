import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json

import cv2

from moviepy.editor import VideoFileClip

import maze.motiondetection as md


####################################################################################################
# scripts setup

sampleVideoPath = "data/videos/"
videoFiles = ["K9", "E7"]

skipFrames = 20

trainingOutput = "data/training"

inlierName = "fish"
outlierName = "background"

winTitle = "Training Frame Selection"

sampleWindow = (128, 128)
frame = None
trackingPoint = None
jumpFrame = False
videoFile = None

good_contours = []
bad_contours = [] 

training_package = {"background": {}, "fish": {}}

# if training package file exists, load it
if os.path.exists("{}/training.json".format(trainingOutput)):
    with open("{}/training.json".format(trainingOutput), "r") as f:
        training_package = json.load(f)

print("Training package: {}".format(training_package))

# motion detection module
motion_detector = md.MotionDetector(10, 3, 15, 400, 3000)

numInliers = len(list(os.listdir(os.path.join(trainingOutput, inlierName))))
numOutliers = len(list(os.listdir(os.path.join(trainingOutput, outlierName))))

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

        imgPath = os.path.join(trainingOutput, inlierName, "sample_{:06d}.png".format(numInliers))
        cv2.imwrite(imgPath, sample)
        numInliers = numInliers + 1
        training_package["fish"][videoFile].append((x, y))
        jumpFrame = True

    elif event == cv2.EVENT_RBUTTONDOWN:
        sample = frame[y-sy2:y+sy2+1, x-sx2:x+sx2+1]

        imgPath = os.path.join(trainingOutput, outlierName, "sample_{:06d}.png".format(numOutliers))
        cv2.imwrite(imgPath, sample)
        numOutliers = numOutliers + 1
        training_package["background"][videoFile].append((x, y))
    else:
        return

    print("Inlier samples: {}, Outlier Samples: {}".format(numInliers, numOutliers))
    #frame_draw = frame.copy()
    #frame_draw = cv2.circle(frame_draw, trackingPoint, sx2, [255, 0, 0])
    #cv2.imshow(winTitle, frame_draw)


videoIndex = 0

while True:
    videoIndex = max(0, min(videoIndex, len(videoFiles)-1))
    videoFile = videoFiles[videoIndex]

    videoPath = sampleVideoPath + videoFile + ".MOV"

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
            cv2.imshow(winTitle, frame)
            return

        prevFrame = video.get_frame((frame_id - 1) / video.fps)
        frame = video.get_frame(frame_id / video.fps)
        nextFrame = video.get_frame((frame_id + 1) / video.fps)

        # to BGR
        prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_RGB2BGR)
        frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)
        nextFrame = cv2.cvtColor(nextFrame, cv2.COLOR_RGB2BGR)

        # calculate motion
        good_contours, bad_contours = motion_detector(prevFrame, frame, nextFrame)

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

        cv2.imshow(winTitle,frame_draw )

    cv2.namedWindow(winTitle, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(winTitle, onMouse)
    cv2.createTrackbar('frame', winTitle, 0, numImages-1, update)

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
            frameIdx = cv2.getTrackbarPos('frame', winTitle)
            frameIdx = max(frameIdx-skipFrames, 0)
            cv2.setTrackbarPos('frame', winTitle, frameIdx)
            update(frameIdx)

        # if arrow right is pressed go to next frame
        if ret == ord('d'):
            frameIdx = cv2.getTrackbarPos('frame', winTitle)
            frameIdx = min(frameIdx+skipFrames, numImages-1)
            cv2.setTrackbarPos('frame', winTitle, frameIdx)
            update(frameIdx)

        if jumpFrame:
            jumpFrame = False
            frameIdx = cv2.getTrackbarPos('frame', winTitle)
            frameIdx = min(frameIdx+1, numImages-1)
            cv2.setTrackbarPos('frame', winTitle, frameIdx)
            update(frameIdx)

    cv2.destroyWindow(winTitle)

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
with open("{}/training.json".format(trainingOutput), "w") as f:
    json.dump(training_package, f, indent=4)
