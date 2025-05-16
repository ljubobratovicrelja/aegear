"""
Utility functions for working with 2D trajectories in image frames,
including drawing, smoothing, and computing properties of motion paths.

Assumes trajectory is a list of (x, y) pixel coordinates sampled at video frame rate.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from scipy.signal import savgol_filter


def smooth_trajectory(trajectory: list[tuple[int, int, int]], filterSize: int = 15) -> list[tuple[int, int, int]]:
    """
    Apply Savitzky-Golay filter to smooth a trajectory.

    Parameters:
        trajectory (list of (t, x, y)): Frame id with raw trajectory points.
        filterSize (int): Window size for filtering (must be odd and >= 5).

    Returns:
        list of (t, x, y): Smoothed trajectory points.
    """
    # Ensure filterSize is odd and at least 5 (polyorder=3, so min window=5)
    if filterSize < 5:
        filterSize = 5
    if filterSize % 2 == 0:
        filterSize += 1
    if len(trajectory) < filterSize:
        return trajectory

    trajectory = np.array(trajectory)
    t = savgol_filter(trajectory[:, 0], filterSize, 3)
    x = savgol_filter(trajectory[:, 1], filterSize, 3)
    y = savgol_filter(trajectory[:, 2], filterSize, 3)

    smoothed = list(zip(t.astype(int), x.astype(int), y.astype(int)))
    return smoothed

def detect_trajectory_outliers(
    trajectory: list[tuple[int, int, int]],
    threshold: float = 20.0  # distance in pixels per frame
) -> list[int]:
    """
    Detects large jumps in pixel space, indicating likely tracking failures.

    Args:
        trajectory: List of (frame_idx, x, y) tuples.
        threshold: Maximum allowed pixel movement per frame.

    Returns:
        List of frame indices where jump exceeds threshold.
    """
    if len(trajectory) < 2:
        return []

    frame_idx, xs, ys = zip(*trajectory)
    xs = np.array(xs)
    ys = np.array(ys)
    frame_idx = np.array(frame_idx)

    dx = np.diff(xs)
    dy = np.diff(ys)
    dist = np.sqrt(dx**2 + dy**2)

    # Mark current frame if jump from previous is too large
    outlier_mask = dist > threshold
    outlier_frames = frame_idx[1:][outlier_mask]  # current frame that made the jump

    return list(outlier_frames)