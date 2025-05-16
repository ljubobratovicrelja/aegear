"""
Utility functions for working with 2D trajectories in image frames,
including drawing, smoothing, and computing properties of motion paths.

Assumes trajectory is a list of (x, y) pixel coordinates sampled at video frame rate.
"""

import numpy as np
from scipy.signal import savgol_filter


def smooth_trajectory(trajectory: list[tuple[int, int, int]], filterSize: int = 15) -> list[tuple[int, int]]:
    """
    Apply Savitzky-Golay filter to smooth a trajectory.

    Parameters:
        trajectory (list of (t, x, y)): Frame id with raw trajectory points.
        filterSize (int): Window size for filtering (must be odd).

    Returns:
        list of (t, x, y): Smoothed trajectory points.
    """
    if len(trajectory) < filterSize:
        return trajectory

    trajectory = np.array(trajectory)
    t = savgol_filter(trajectory[:, 0], filterSize, 3)
    x = savgol_filter(trajectory[:, 1], filterSize, 3)
    y = savgol_filter(trajectory[:, 2], filterSize, 3)

    smoothed = list(zip(t.astype(int), x.astype(int), y.astype(int)))
    return smoothed
