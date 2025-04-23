"""
Utility functions for working with 2D trajectories in image frames,
including drawing, smoothing, and computing properties of motion paths.

Assumes trajectory is a list of (x, y) pixel coordinates sampled at video frame rate.
"""

import cv2
import numpy as np
from scipy.signal import savgol_filter


def trajectory_length(trajectory: list[tuple[int, int]]) -> float:
    """
    Compute the total length of a 2D trajectory.

    Parameters:
        trajectory (list of (x, y)): Sequence of points.

    Returns:
        float: Sum of Euclidean distances between points.
    """
    if len(trajectory) < 2:
        return 0.0

    prev = trajectory[0]
    sumlength = 0.0
    for p in trajectory[1:]:
        sumlength += np.linalg.norm(np.array(p) - np.array(prev))
        prev = p

    return sumlength


def smooth_trajectory(trajectory: list[tuple[int, int]], filterSize: int = 15) -> list[tuple[int, int]]:
    """
    Apply Savitzky-Golay filter to smooth a trajectory.

    Parameters:
        trajectory (list of (x, y)): Raw trajectory points.
        filterSize (int): Window size for filtering (must be odd).

    Returns:
        list of (x, y): Smoothed trajectory points.
    """
    if len(trajectory) < filterSize:
        return trajectory

    trajectory = np.array(trajectory)
    x = savgol_filter(trajectory[:, 0], filterSize, 3)
    y = savgol_filter(trajectory[:, 1], filterSize, 3)

    smoothed = list(zip(x.astype(int), y.astype(int)))
    return smoothed

def draw_trajectory(
    frame: np.ndarray,
    trajectory: list[tuple[int, int]],
    thickness: int = 1,
    n_seconds: float = 3.0,
    fps: float = 60.0
) -> np.ndarray:
    """
    Draw a trajectory onto a video frame, with optional fading over time.

    The trajectory is clipped to the last `n_seconds` of data. The oldest points
    fade out linearly, and the newest are fully opaque.

    Parameters:
        frame (np.ndarray): Input video frame (BGR).
        trajectory (list of (x, y)): Sequence of 2D points.
        thickness (int): Line thickness.
        n_seconds (float): Time window to draw (in seconds).
        fps (float): Frame rate to interpret the trajectory timing.

    Returns:
        np.ndarray: Frame with trajectory overlay.
    """
    if len(trajectory) < 2:
        return frame

    max_points = int(n_seconds * fps)
    trajectory = trajectory[-max_points:]
    num_points = len(trajectory)

    dframe = np.copy(frame)
    overlay = dframe.copy()

    # Precompute colors and alpha values
    hsv_colors = np.zeros((num_points, 1, 3), dtype=np.uint8)
    alphas = np.linspace(0.0, 1.0, num_points)
    for i in range(num_points):
        hsv_colors[i, 0, 0] = int(150.0 * ((num_points - i) / num_points))  # H
        hsv_colors[i, 0, 1] = 255  # S
        hsv_colors[i, 0, 2] = 255  # V
    rgb_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2RGB).reshape(num_points, 3)

    for i in range(1, num_points):
        prev = trajectory[i - 1]
        p = trajectory[i]
        color = tuple(int(c) for c in rgb_colors[i])
        alpha = alphas[i]

        # Determine bounding box for the line
        x1, y1 = prev
        x2, y2 = p
        x_min = max(min(x1, x2) - thickness, 0)
        y_min = max(min(y1, y2) - thickness, 0)
        x_max = min(max(x1, x2) + thickness, frame.shape[1] - 1)
        y_max = min(max(y1, y2) + thickness, frame.shape[0] - 1)

        roi_overlay = overlay[y_min:y_max+1, x_min:x_max+1]
        roi_temp = roi_overlay.copy()

        # Draw line in the ROI
        cv2.line(roi_temp, (x1 - x_min, y1 - y_min), (x2 - x_min, y2 - y_min), color, thickness, cv2.LINE_AA)
        cv2.addWeighted(roi_temp, alpha, roi_overlay, 1 - alpha, 0, roi_overlay)

        overlay[y_min:y_max+1, x_min:x_max+1] = roi_overlay

    return overlay