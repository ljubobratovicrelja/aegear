#!/usr/bin/env python3
"""
Camera calibration script.

This script calibrates a camera using a chessboard pattern and saves the calibration parameters
to an OpenCV file. These parameters can be used in another script to undistort images.

The configuration descriptor is a JSON file with the following fields:
    - videoPath: path to the video file
    - cornerDataPath: (optional) path to save/load the corner data file
    - calibrationDataPath: path to save the calibration data
    - patternSize: [rows, columns] number of inner chessboard corners (pattern)
    - subPixWinSize: window size for sub-pixel refinement (e.g. [5, 5] for HD resolution)
    - patternSquareSideLength: length of one square side in cm (default provided if missing)
    - skipFrames: number of frames to skip when sampling from the video
"""

import sys
import os
import json

import numpy as np
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

from aegear.video import VideoClip


def load_descriptor(path: str) -> dict:
    """
    Load calibration configuration from a JSON descriptor file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict: Descriptor data.
    """
    with open(path, "r") as f:
        descriptor = json.load(f)
    return descriptor


def get_video_metadata(video_path: str):
    """
    Load the video and compute its total number of frames.

    Args:
        video_path (str): Path to the video file.

    Returns:
        tuple: (VideoClip object, number of frames)
    """
    video = VideoClip(video_path)
    num_frames = int(video.fps * video.duration)
    return video, num_frames


def detect_corners(video: VideoClip, num_frames: int, pattern_size, sub_pix_win_size, criteria, skip_frames: int):
    """
    Process video frames and detect chessboard corners.

    Args:
        video (VideoClip): Video clip object.
        num_frames (int): Total number of frames in the video.
        pattern_size (tuple): Number of inner corners per row and column.
        sub_pix_win_size (tuple): Window size for sub-pixel refinement.
        criteria (tuple): Termination criteria for the cornerSubPix algorithm.
        skip_frames (int): Number of frames to skip between samples.

    Returns:
        tuple: (list of valid frames, numpy array of detected corners)
    """
    valid_frames = []
    all_corners = []  # Collection of detected corners for calibration

    for frame_id in tqdm(range(0, num_frames, skip_frames), desc="Processing frames"):
        # Extract frame at the specified time
        frame = video.get_frame(float(frame_id) / video.fps)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Use a fast check to detect the chessboard pattern
        found, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_FAST_CHECK)
        if found:
            # Refine corner locations to sub-pixel accuracy
            corners = cv2.cornerSubPix(gray, corners, sub_pix_win_size, (-1, -1), criteria)
            all_corners.append(corners)
            valid_frames.append(gray)  # Save grayscale frame for later undistortion check

    if all_corners:
        all_corners = np.array(all_corners)
    return valid_frames, all_corners


def save_corners(corner_data_path: str, corners: np.ndarray):
    """
    Save detected corners to disk.

    Args:
        corner_data_path (str): Path to save the corner data.
        corners (np.ndarray): Array of detected corners.
    """
    try:
        np.save(corner_data_path, corners)
        print(f"Saved corners to {corner_data_path}")
    except Exception as e:
        print(f"Failed to save corners to {corner_data_path}: {e}")


def load_corners(corner_data_path: str) -> np.ndarray:
    """
    Load corner data from disk.

    Args:
        corner_data_path (str): Path from where to load the corner data.

    Returns:
        np.ndarray: Loaded corners.
    """
    return np.load(corner_data_path)


def generate_object_points(pattern_size, pattern_square_side_length, num_samples: int):
    """
    Generate object points for calibration (3D real-world coordinates).

    Args:
        pattern_size (tuple): Number of inner corners per row and column.
        pattern_square_side_length (float): Length of one square side in cm.
        num_samples (int): Number of calibration images/samples.

    Returns:
        np.ndarray: Object points in the required shape and type.
    """
    # Create a single grid of 3D points for the chessboard pattern (z=0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= pattern_square_side_length

    # Duplicate the grid for each calibration sample
    obj_points = np.array([objp] * num_samples)
    obj_points = np.reshape(obj_points, (num_samples, objp.shape[0], 1, 3))
    if obj_points.dtype != np.float32 or obj_points.shape[-1] != 3:
        obj_points = obj_points.astype(np.float32)
    return obj_points


def calibrate_camera(obj_points, img_points, image_size):
    """
    Calibrate the camera using the object and image points.

    Args:
        obj_points (np.ndarray): 3D points in real-world space.
        img_points (np.ndarray): 2D points in image plane.
        image_size (tuple): Size of the image (width, height).

    Returns:
        tuple: Calibration result, camera matrix, distortion coefficients,
               rotation vectors, and translation vectors.
    """
    return cv2.calibrateCamera(obj_points, img_points, image_size, None, None)


def save_calibration(calibration_data_path: str, camera_matrix, distortion_coeffs):
    """
    Save the calibration parameters using OpenCV's FileStorage.

    Args:
        calibration_data_path (str): File path to save calibration data.
        camera_matrix: Camera matrix.
        distortion_coeffs: Distortion coefficients.
    """
    storage = cv2.FileStorage(calibration_data_path, cv2.FILE_STORAGE_WRITE)
    storage.write("mtx", camera_matrix)
    storage.write("dist", distortion_coeffs)
    storage.release()
    print(f"Saved calibration data to {calibration_data_path}")


def undistort_image(image, camera_matrix, distortion_coeffs):
    """
    Undistort an image using the calibration parameters.

    Args:
        image (np.ndarray): Original distorted image.
        camera_matrix: Camera matrix from calibration.
        distortion_coeffs: Distortion coefficients.

    Returns:
        np.ndarray: Undistorted image.
    """
    return cv2.undistort(image, camera_matrix, distortion_coeffs)


def display_undistortion(original_img, undistorted_img):
    """
    Display the original and undistorted images side by side for visual comparison.

    Args:
        original_img (np.ndarray): Original image.
        undistorted_img (np.ndarray): Image after undistortion.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_img, cmap="gray")
    ax[0].set_title("Original")
    ax[1].imshow(undistorted_img, cmap="gray")
    ax[1].set_title("Undistorted")
    plt.show()


def main():
    # Ensure proper usage
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {sys.argv[0]} <path_to_descriptor>")

    # Load configuration
    descriptor_path = sys.argv[1]
    descriptor = load_descriptor(descriptor_path)
    
    video_path = descriptor["videoPath"]
    corner_data_path = descriptor.get("cornerDataPath")
    calibration_data_path = descriptor["calibrationDataPath"]
    pattern_size = tuple(descriptor["patternSize"])
    sub_pix_win_size = tuple(descriptor["subPixWinSize"])
    pattern_square_side_length = descriptor.get("patternSquareSideLength", 1.6)
    skip_frames = descriptor.get("skipFrames", 120)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Load video and compute metadata
    video, num_frames = get_video_metadata(video_path)
    print(f"Video has {num_frames} frames. Sampling every {skip_frames}th frame, "
          f"processing approximately {num_frames // skip_frames} frames.")

    # Read existing corner data if available, otherwise detect from video
    if corner_data_path and os.path.exists(corner_data_path):
        print("Loading existing corner data.")
        all_corners = load_corners(corner_data_path)
        # For undistortion demonstration, still extract valid frames
        valid_frames, _ = detect_corners(video, num_frames, pattern_size, sub_pix_win_size, criteria, skip_frames)
    else:
        print("Detecting corners from video frames.")
        valid_frames, all_corners = detect_corners(video, num_frames, pattern_size, sub_pix_win_size, criteria, skip_frames)
        if corner_data_path:
            save_corners(corner_data_path, all_corners)
        else:
            print("Corner data will not be saved (no cornerDataPath provided).")
    
    num_samples = all_corners.shape[0]
    print(f"Collected {num_samples} corner samples.")

    # Generate object points corresponding to the detected corners
    obj_points = generate_object_points(pattern_size, pattern_square_side_length, num_samples)

    if not valid_frames:
        sys.exit("No valid frames were found for calibration.")

    # Use the first valid frame to determine the image size (width, height)
    frame_height, frame_width = valid_frames[0].shape[:2]
    image_size = (frame_width, frame_height)

    print("Running calibration...")
    ret, camera_matrix, distortion_coeffs, rvecs, tvecs = calibrate_camera(obj_points, all_corners, image_size)

    if ret:
        print("Calibration successful.")
        print(f"Camera Matrix:\n{camera_matrix}")
        print(f"Distortion Coefficients:\n{distortion_coeffs}")
        save_calibration(calibration_data_path, camera_matrix, distortion_coeffs)
    else:
        sys.exit("Calibration failed.")

    # Demonstrate undistortion on a sample image
    original_img = valid_frames[0].copy()
    undistorted_img = undistort_image(original_img, camera_matrix, distortion_coeffs)
    display_undistortion(original_img, undistorted_img)


if __name__ == "__main__":
    main()
