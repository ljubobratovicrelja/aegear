import cv2
import pickle
import numpy as np


def readfirstframe(vidpath, offset=100):
    videoStream = cv2.VideoCapture(vidpath)
    assert videoStream.isOpened(), "Failed opening video at {}".format(vidpath)

    while True:
        ret, frame = videoStream.read()
        if ret:
            offset = offset - 1
            if offset == 0:
                videoStream.release()
                return frame


mazedesignpath = "..\data\maze_design.png"
mazesamplevideopath = "../data/videos/EE4.MOV"

mazepointspath = "..\data\mazemodel.pickle"
calibpath = "..\data\calibration.xml"

mazedesignimg = cv2.imread(mazedesignpath)
mazesampleimg = readfirstframe(mazesamplevideopath)

# read calibration to undistort

storage = cv2.FileStorage(calibpath, cv2.FILE_STORAGE_READ)
mtx = storage.getNode("mtx").mat()
dist = storage.getNode("dist").mat()

mazesampleimg = cv2.undistort(mazesampleimg, mtx, dist, None, None)

MAZE_DESIGN_WINTITLE = "Maze Design"
MAZE_IMAGE_WINTITLE = "Maze Sample"

mazepoints = []



def redrawmaze():
    global mazepoints
    drawimg = np.copy(mazesampleimg)

    radius = int(max(5, (drawimg.shape[0] + drawimg.shape[1]) * 0.5 * 0.005))

    for p in mazepoints:
        cv2.circle(drawimg, p, radius, (255, 255, 255), 2, cv2.LINE_AA)

    if len(mazepoints) >= 2:
        p = np.array(mazepoints, np.int32).reshape((-1, 1, 2))
        cv2.polylines(drawimg, [p], False, [255, 0, 255], 2, cv2.LINE_AA)

    cv2.imshow(MAZE_IMAGE_WINTITLE, drawimg)

def drawfullmaze():
    global mazepoints
    assert len(mazepoints) == 28, "Maze is not full"

    drawimg = np.copy(mazesampleimg)

    b1 = np.array([mazepoints[0], mazepoints[1], mazepoints[26], mazepoints[27]], np.int32).reshape((-1,1,2))
    b2 = np.array([mazepoints[5], mazepoints[6], mazepoints[7], mazepoints[8]], np.int32).reshape((-1,1,2))
    b3 = np.array([mazepoints[12], mazepoints[13], mazepoints[14], mazepoints[15]], np.int32).reshape((-1,1,2))
    b4 = np.array([mazepoints[19], mazepoints[20], mazepoints[21], mazepoints[22]], np.int32).reshape((-1,1,2))

    cv2.polylines(drawimg, [b1], True, [255, 0, 0], 2, cv2.LINE_AA)
    cv2.polylines(drawimg, [b2], True, [0, 255, 0], 2, cv2.LINE_AA)
    cv2.polylines(drawimg, [b3], True, [0, 0, 255], 2, cv2.LINE_AA)
    cv2.polylines(drawimg, [b4], True, [255, 0, 255], 2, cv2.LINE_AA)

    cv2.imshow(MAZE_IMAGE_WINTITLE, drawimg)


def redrawmazedesign():
    global mazepoints
    cv2.imshow(MAZE_DESIGN_WINTITLE, mazedesignimg)


def selectMazePoints(event, x, y, flags, params):
    global mazepoints
    if event == cv2.EVENT_LBUTTONDOWN:  # click, add point
        mazepoints.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and mazepoints:
        mazepoints.pop()


cv2.namedWindow(MAZE_DESIGN_WINTITLE, cv2.WINDOW_NORMAL)
cv2.namedWindow(MAZE_IMAGE_WINTITLE, cv2.WINDOW_NORMAL)

cv2.setMouseCallback(MAZE_IMAGE_WINTITLE, selectMazePoints, None)

while True:
    redrawmaze()
    redrawmazedesign()

    if cv2.waitKey(1) == ord('q'):
        break

    if len(mazepoints) == 28:
        print("Full maze!")
        break

drawfullmaze()
cv2.waitKey()



