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
    threshold: float = 3.0,
    window: int = 5
) -> list[int]:
    """
    Vectorized outlier detection using rolling median and MAD.

    Args:
        trajectory: List of (frame_idx, x, y) tuples.
        threshold: Z-score threshold for MAD.
        window: Half-window size for rolling stats.

    Returns:
        List of frame indices marked as outliers.
    """
    if len(trajectory) < 2 * window + 1:
        return []

    frame_idx, xs, ys = zip(*trajectory)
    xs = np.array(xs)
    ys = np.array(ys)
    n = len(xs)

    # Create sliding windows over x and y
    x_windows = sliding_window_view(xs, 2 * window + 1)
    y_windows = sliding_window_view(ys, 2 * window + 1)

    # Medians and MADs
    x_medians = np.median(x_windows, axis=1)
    y_medians = np.median(y_windows, axis=1)
    x_mads = np.median(np.abs(x_windows - x_medians[:, None]), axis=1) + 1e-6
    y_mads = np.median(np.abs(y_windows - y_medians[:, None]), axis=1) + 1e-6

    # Compute Z-scores for center points
    center_indices = np.arange(window, n - window)
    x_center = xs[center_indices]
    y_center = ys[center_indices]
    z_x = np.abs(x_center - x_medians) / x_mads
    z_y = np.abs(y_center - y_medians) / y_mads

    outliers = (z_x > threshold) | (z_y > threshold)
    return list(np.array(frame_idx)[center_indices[outliers]])