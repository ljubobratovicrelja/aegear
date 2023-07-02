import os

import cv2
import pickle

from tqdm import tqdm


####################################################################################################
# scripts setup

sampleVideoPath = "data/videos/"
videoFiles = ["K9","EE1", "S1"]

skipFrames = 5
motionThreshold = 10

trainingOutput = "data/training"

inlierName = "fish"
outlierName = "background"

winTitle = "Training Frame Selection"

sampleWindow = (32, 32)
frame = None
trackingPoint = None
jumpFrame = False

numInliers = len(list(os.listdir(os.path.join(trainingOutput, inlierName))))
numOutliers = len(list(os.listdir(os.path.join(trainingOutput, outlierName))))

print("Inlier samples: {}, Outlier Samples: {}".format(numInliers, numOutliers))


def onMouse(event, x, y, flags, param):
    global trackingPoint
    global frame
    global numInliers
    global numOutliers
    global jumpFrame

    # store point for drawing
    trackingPoint = (x, y)

    # sample image window
    sx2 = int(sampleWindow[0]/2)
    sy2 = int(sampleWindow[1]/2)
    sample = frame[y-sy2:y+sy2+1, x-sx2:x+sx2+1]

    if event == cv2.EVENT_LBUTTONDOWN:
        imgPath = os.path.join(trainingOutput, inlierName, "sample_{:06d}.png".format(numInliers))
        print(imgPath)
        cv2.imwrite(imgPath, sample)
        numInliers = numInliers + 1
        jumpFrame = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        imgPath = os.path.join(trainingOutput, outlierName, "sample_{:06d}.png".format(numOutliers))
        print(imgPath)
        cv2.imwrite(imgPath, sample)
        numOutliers = numOutliers + 1
    else:
        return

    print("Inlier samples: {}, Outlier Samples: {}".format(numInliers, numOutliers))
    frame_draw = frame.copy()
    frame_draw = cv2.circle(frame_draw, trackingPoint, sx2, [255, 0, 0])
    cv2.imshow(winTitle, frame_draw)


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

