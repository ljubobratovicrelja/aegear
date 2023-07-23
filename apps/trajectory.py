import cv2
import numpy as np
from scipy.signal import savgol_filter


def inBounds(cx, cy, s2):
    global IMROWS
    global IMCOLS

    p1 = (cy - s2, cx - s2)
    p2 = (cy + s2, cx + s2)

    if p1[0] < 0 or p1[1] < 0:
        return False
    if p2[0] >= IMROWS or p2[1] >= IMCOLS:
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


def smoothTrajectory(trajectory, filterSize=15):
    if len(trajectory) < filterSize:
        return trajectory

    trajectory = np.array(trajectory)

    x = trajectory[:, 0]
    y = trajectory[:, 1]

    x = savgol_filter(x, filterSize, 3)
    y = savgol_filter(y, filterSize, 3)

    x = list(map(lambda v: int(v), x))
    y = list(map(lambda v: int(v), y))

    trajectory = list(zip(x, y))

    return trajectory

def drawTrajectory(frame, trajectory, thickness=1):
    if len(trajectory) < 2:
        return frame

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