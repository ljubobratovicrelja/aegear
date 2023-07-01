import os

import cv2
import pickle

from tqdm import tqdm


####################################################################################################
# scripts setup

sampleVideoPath = "data/videos/"
videoFiles = ["EE1", "K9", "S1"]

modelPath =  "data/svc.pickle"
dataPath = "data/training.npy"
startFrame = 1000
skipFrames = 10
motionThreshold = 10

trainingOutput = "data/videos/training"

inlierName = "inlier"
outlierName = "outlier"

winTitle = "Training Frame Selection"

sampleWindow = (32, 32)
frame = None
trackingPoint = None

numInliers = len(list(filter(lambda f: inlierName in f, os.listdir(trainingOutput))))
numOutliers = len(list(filter(lambda f: outlierName in f, os.listdir(trainingOutput))))

print("Inlier samples: {}, Outlier Samples: {}".format(numInliers, numOutliers))


def onMouse(event, x, y, flags, param):
    global trackingPoint
    global frame
    global numInliers
    global numOutliers

    # store point for drawing
    trackingPoint = (x, y)

    # sample image window
    sx2 = int(sampleWindow[0]/2)
    sy2 = int(sampleWindow[1]/2)
    sample = frame[y-sy2:y+sy2+1, x-sx2:x+sx2+1]

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.imwrite(trainingOutput + "/{}_{:06d}.png".format(inlierName, numInliers), sample)
        numInliers = numInliers + 1
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.imwrite(trainingOutput + "/{}_{:06d}.png".format(outlierName, numOutliers), sample)
        numOutliers = numOutliers + 1
    else:
        return

    print("Inlier samples: {}, Outlier Samples: {}".format(numInliers, numOutliers))
    frame = cv2.circle(frame, trackingPoint, sx2, [255, 0, 0])
    cv2.imshow(winTitle, frame)


videoIndex = 0
while True:
    videoIndex = max(0, min(videoIndex, len(videoFiles)-1))
    videoFile = videoFiles[videoIndex]

    videoPath = sampleVideoPath + videoFile

    print("Reading video {}...".format(videoFile))
    imageFiles = list(filter(lambda f: ".png" in f, os.listdir(videoPath)))
    images = []

    for imageFile in tqdm(imageFiles):
        impath = videoPath + "/" + imageFile
        images.append(cv2.imread(impath))

    goToNextVideo = False
    goToPrevVideo = False
    quitReading = False
        
    numImages = len(images)

    print("Done. Showing frames...")

    def update(val):
        global frame
        frame = images[val]
        cv2.imshow(winTitle, frame)

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

