#!/usr/bin/env python3
"""
Extracts training data from videos for the motion detection module.

This script allows the user to select inlier samples ("fish") using left-clicks and outlier
samples ("background") using right-clicks on video frames. It also supports navigation between
video files and frames. Extracted samples are saved to a specified directory and a training package
file is maintained.

Usage:
    python trainingdataextraction.py <data_collection_descriptor>.json

Data Collection Descriptor JSON must include:
    - videosRoot: Directory containing video files.
    - videoFiles: List of video file names.
    - skipFrames: Number of frames to skip for navigation.
    - trainingDataRoot: Directory for saving training samples.
    - sampleWindow: Dimension (in pixels) for the square sample window.
    
Example JSON:
{
    "videosRoot": "C:/Users/JohnDoe/Desktop/videos/",
    "videoFiles": ["video1", "video2", "video3"],
    "skipFrames": 10,
    "trainingDataRoot": "C:/Users/JohnDoe/Desktop/trainingdata/",
    "sampleWindow": 64
}
"""

import os
import sys
import json
import cv2
from moviepy.editor import VideoFileClip
import aegear.motiondetection as md

# Categories for sample classification and display window title.
INLIER_NAME = "fish"
OUTLIER_NAME = "background"
WIN_TITLE = "Training Frame Selection"


def print_welcome_screen():
    """
    Display a welcome screen with key mappings on the command line.
    """
    welcome_text = """
    ============================================
           Training Data Extraction Tool
    ============================================
    Key Bindings:
      q : Quit the program
      n : Next video
      p : Previous video
      a : Go back by skipFrames frames
      d : Advance by skipFrames frames
      Left Mouse Click  : Extract inlier sample ("fish")
      Right Mouse Click : Extract outlier sample ("background")
    ============================================
    """
    print(welcome_text)


def load_descriptor(descriptor_path):
    """
    Load the data collection descriptor from the specified JSON file.
    
    Raises:
        SystemExit: If the file does not exist or is empty.
    """
    if not os.path.exists(descriptor_path):
        sys.exit(f"Data collection descriptor {descriptor_path} does not exist")
    with open(descriptor_path, 'r') as f:
        descriptor = json.load(f)
    if not descriptor:
        sys.exit(f"Data collection descriptor {descriptor_path} is empty")
    return descriptor


def load_training_package(training_data_root):
    """
    Load the training package if it exists; otherwise, return a new package structure.
    
    The package is expected to have the keys 'background' and 'fish'.
    """
    training_package_path = os.path.join(training_data_root, "training.json")
    if os.path.exists(training_package_path):
        with open(training_package_path, "r") as f:
            return json.load(f)
    return {"background": {}, "fish": {}}


# Global variables used for frame processing and callbacks.
frame = None
trackingPoint = None
jumpFrame = False
videoFile = None
good_contours = []
bad_contours = []


def onMouse(event, x, y, flags, param):
    """
    Callback for mouse events to extract samples.

    Left-click:
      - If the click is within any detected contour, use the contour's center as the sample center.
      - Save the sample as an inlier.
      - Set a flag to advance to the next frame.
      
    Right-click:
      - Save the sample at the click location as an outlier.
    """
    global trackingPoint, frame, jumpFrame, videoFile, good_contours, bad_contours
    global numInliers, numOutliers, training_package

    # Calculate half the dimensions of the sample window.
    sx2 = sampleWindow[0] // 2
    sy2 = sampleWindow[1] // 2

    trackingPoint = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        for contour in (good_contours + bad_contours):
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    x = int(M['m10'] / M['m00'])
                    y = int(M['m01'] / M['m00'])
                    trackingPoint = (x, y)
                break

        sample = frame[y - sy2 : y + sy2 + 1, x - sx2 : x + sx2 + 1]
        imgPath = os.path.join(trainingDataRoot, INLIER_NAME, f"sample_{numInliers:06d}.png")
        cv2.imwrite(imgPath, sample)
        numInliers += 1
        training_package["fish"][videoFile].append((x, y))
        jumpFrame = True

    elif event == cv2.EVENT_RBUTTONDOWN:
        sample = frame[y - sy2 : y + sy2 + 1, x - sx2 : x + sx2 + 1]
        imgPath = os.path.join(trainingDataRoot, OUTLIER_NAME, f"sample_{numOutliers:06d}.png")
        cv2.imwrite(imgPath, sample)
        numOutliers += 1
        training_package["background"][videoFile].append((x, y))

    print(f"Inlier samples: {numInliers}, Outlier samples: {numOutliers}")


def update(frame_id, video, fps):
    """
    Refresh the displayed frame and process motion detection.

    For frame_id == 0, display the initial frame. For subsequent frames, retrieve
    adjacent frames, perform motion detection, and overlay detected contours with markers.
    """
    global frame, good_contours, bad_contours

    if frame_id == 0:
        frame = video.get_frame(0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(WIN_TITLE, frame)
        return

    prev_frame = cv2.cvtColor(video.get_frame((frame_id - 1) / fps), cv2.COLOR_RGB2BGR)
    frame = cv2.cvtColor(video.get_frame(frame_id / fps), cv2.COLOR_RGB2BGR)
    next_frame = cv2.cvtColor(video.get_frame((frame_id + 1) / fps), cv2.COLOR_RGB2BGR)

    good_contours, bad_contours = motion_detector.detect(prev_frame, frame, next_frame)

    frame_draw = frame.copy()
    cv2.drawContours(frame_draw, good_contours, 0, [0, 255, 0], 2)
    cv2.drawContours(frame_draw, bad_contours, 0, [0, 0, 255], 2)

    for contour in good_contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.drawMarker(frame_draw, (cx, cy), [255, 255, 255], cv2.MARKER_CROSS, 10, 2)

    cv2.imshow(WIN_TITLE, frame_draw)


def main():
    global training_package, numInliers, numOutliers, videoFile, jumpFrame
    global sampleWindow, trainingDataRoot, motion_detector

    print_welcome_screen()

    if len(sys.argv) < 2:
        sys.exit("Usage: python trainingdataextraction.py <data_collection_descriptor>.json")

    descriptor = load_descriptor(sys.argv[1])
    videosRoot = descriptor["videosRoot"]
    videoFiles = descriptor["videoFiles"]
    skipFrames = descriptor["skipFrames"]
    trainingDataRoot = descriptor["trainingDataRoot"]
    sample_size = descriptor["sampleWindow"]
    sampleWindow = (sample_size, sample_size)

    motion_detector = md.MotionDetector(10, 3, 15, 400, 3000)

    for category in [INLIER_NAME, OUTLIER_NAME]:
        os.makedirs(os.path.join(trainingDataRoot, category), exist_ok=True)

    training_package = load_training_package(trainingDataRoot)

    numInliers = len(os.listdir(os.path.join(trainingDataRoot, INLIER_NAME)))
    numOutliers = len(os.listdir(os.path.join(trainingDataRoot, OUTLIER_NAME)))
    print(f"Inlier samples: {numInliers}, Outlier samples: {numOutliers}")

    videoIndex = 0

    while True:
        videoIndex = max(0, min(videoIndex, len(videoFiles) - 1))
        videoFile = videoFiles[videoIndex]
        videoPath = os.path.join(videosRoot, videoFile)

        if not os.path.exists(videoPath):
            print(f"Video file {videoPath} does not exist")
            videoIndex += 1
            continue

        if videoFile not in training_package["background"]:
            training_package["background"][videoFile] = []
        if videoFile not in training_package["fish"]:
            training_package["fish"][videoFile] = []

        video = VideoFileClip(videoPath)
        fps = video.fps
        numImages = int(video.duration * fps)

        goToNextVideo = False
        goToPrevVideo = False
        quitReading = False

        cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN_TITLE, onMouse)
        cv2.createTrackbar('frame', WIN_TITLE, 0, numImages - 1,
                           lambda pos: update(pos, video, fps))
        update(0, video, fps)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                quitReading = True
                break
            if key == ord('n'):
                goToNextVideo = True
                break
            if key == ord('p'):
                goToPrevVideo = True
                break

            if key == ord('a'):
                frameIdx = cv2.getTrackbarPos('frame', WIN_TITLE)
                frameIdx = max(frameIdx - skipFrames, 0)
                cv2.setTrackbarPos('frame', WIN_TITLE, frameIdx)
                update(frameIdx, video, fps)

            if key == ord('d'):
                frameIdx = cv2.getTrackbarPos('frame', WIN_TITLE)
                frameIdx = min(frameIdx + skipFrames, numImages - 1)
                cv2.setTrackbarPos('frame', WIN_TITLE, frameIdx)
                update(frameIdx, video, fps)

            if jumpFrame:
                jumpFrame = False
                frameIdx = cv2.getTrackbarPos('frame', WIN_TITLE)
                frameIdx = min(frameIdx + 1, numImages - 1)
                cv2.setTrackbarPos('frame', WIN_TITLE, frameIdx)
                update(frameIdx, video, fps)

        cv2.destroyWindow(WIN_TITLE)

        if goToNextVideo:
            videoIndex = min(videoIndex + 1, len(videoFiles) - 1)
            continue
        if goToPrevVideo:
            videoIndex = max(videoIndex - 1, 0)
            continue
        if quitReading:
            break

    cv2.destroyAllWindows()

    training_package_path = os.path.join(trainingDataRoot, "training.json")
    with open(training_package_path, "w") as f:
        json.dump(training_package, f, indent=4)


if __name__ == '__main__':
    main()
