"""
Utility functions for working with 2D trajectories in image frames,
including drawing, smoothing, and computing properties of motion paths.

Assumes trajectory is a list of (x, y) pixel coordinates sampled at video frame rate.
"""

import numpy as np
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
    Detect outlier frames in a trajectory based on local standard deviation.
    For each point, if its distance from the local mean (in a window) exceeds
    threshold * local std, it is considered an outlier.

    Parameters:
        trajectory: list of (frame, x, y)
        threshold: float, number of std deviations to consider as outlier
        window: int, size of the local window (must be odd, default 5)
    Returns:
        List of frame indices (ints) that are outliers.
    """
    if len(trajectory) < window or window < 3:
        return []
    if window % 2 == 0:
        window += 1
    half = window // 2
    traj_arr = np.array(trajectory)
    outlier_frames = []
    for i in range(len(traj_arr)):
        start = max(0, i - half)
        end = min(len(traj_arr), i + half + 1)
        local = traj_arr[start:end, 1:3]  # x, y only
        if len(local) < 3:
            continue
        mean = np.mean(local, axis=0)
        std = np.std(local, axis=0)
        dist = np.linalg.norm(traj_arr[i, 1:3] - mean)
        std_total = np.linalg.norm(std)
        if std_total > 0 and dist > threshold * std_total:
            outlier_frames.append(int(traj_arr[i, 0]))
    return outlier_frames
