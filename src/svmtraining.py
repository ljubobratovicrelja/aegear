import cv2
import sklearn.svm as svm
import pickle


####################################################################################################
# scripts setup

sampleVideoPath = "data/videos/EE1.MOV"
modelPath=  "data/svc.pickle"
startFrame = 3000
skipFrames = 30
motionThreshold = 10
winTitle = "Training Frame Selection"

videoStream = cv2.VideoCapture(sampleVideoPath)

assert videoStream.isOpened(), "Failed opening video at {}".format(sampleVideoPath)

sampleWindow = (32, 32)
frame = None
trackingPoint = None
trainingPositiveSamples = []
trainingNegativeSamples = []


def onMouse(event, x, y, flags, param):
    global trackingPoint
    global frame

    if event == cv2.EVENT_LBUTTONDOWN:
        # store point for drawing
        trackingPoint = (x, y)

        # sample image window
        sx2 = int(sampleWindow[0]/2)
        sy2 = int(sampleWindow[1]/2)
        trainingPositiveSamples.append(frame[y-sy2:y+sy2+1, x-sx2:x+sx2+1].copy())

        frame = cv2.circle(frame, trackingPoint, sx2, [255, 0, 0])
        cv2.imshow(winTitle, frame)

    if event == cv2.EVENT_RBUTTONDOWN:
        # store point for drawing
        trackingPoint = (x, y)

        # sample image window
        sx2 = int(sampleWindow[0]/2)
        sy2 = int(sampleWindow[1]/2)
        trainingNegativeSamples.append(frame[y-sy2:y+sy2+1, x-sx2:x+sx2+1].copy())

        frame = cv2.circle(frame, trackingPoint, sx2, [255, 0, 0])
        cv2.imshow(winTitle, frame)


cv2.namedWindow(winTitle, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(winTitle, onMouse)

while True:
    ret, frame = videoStream.read()
    trackingPoint = None
    if not ret:
        break

    if startFrame > 0:
        startFrame -= 1
        continue

    quitReading = False
    for skipFrame in range(skipFrames):
        ret, frame = videoStream.read()
        if not ret:
            quitReading = True
            break

    if quitReading:
        break

    print("Positive samples: {}, Negative Samples: {}".format(len(trainingPositiveSamples), len(trainingNegativeSamples)))
    while True:
        cv2.imshow(winTitle, frame)
        ret = cv2.waitKey(50)

        if ret == ord('q'):
            quitReading = True
            break
        if ret == 32:
            break  # just skip this frame, its no good for training

        if trackingPoint is not None:
            break

    if quitReading:
        break

videoStream.release()
cv2.destroyAllWindows()

# description
hog = cv2.HOGDescriptor(sampleWindow, (16, 16), (8, 8), (16, 16), 8)

print("Describing features...")
descriptionsPositive = []  # list(map(lambda d: hog.compute(d), trainingPositiveSamples))
descriptionsNegative = []  # list(map(lambda d: hog.compute(d), trainingNegativeSamples))

for i, s in enumerate(trainingPositiveSamples):
    try:
        descriptionsPositive.append(hog.compute(s))
    except:
        print("Failed describing sample no {}".format(i))

for i, s in enumerate(trainingNegativeSamples):
    try:
        descriptionsNegative.append(hog.compute(s))
    except:
        print("Failed describing sample no {}".format(i))

print("done")

X = descriptionsPositive + descriptionsNegative
Y = [1] * len(descriptionsPositive) + [0] * len(descriptionsNegative)

# svm training
svc = svm.SVC()
svc.fit(X, Y)

print("Writing model pickle to: {}".format(modelPath))
with open(modelPath, 'wb') as f:
    pickle.dump(svc, f)

