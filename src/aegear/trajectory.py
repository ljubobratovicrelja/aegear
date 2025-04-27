"""
Utility functions for working with 2D trajectories in image frames,
including drawing, smoothing, and computing properties of motion paths.

Assumes trajectory is a list of (x, y) pixel coordinates sampled at video frame rate.
"""

import cv2
import numpy as np
from scipy.signal import savgol_filter
import bisect
from typing import List, Tuple


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

def draw_trajectory(
    frame: np.ndarray,
    trajectory: List[Tuple[int, int, int]],
    current_t: int,
    n_seconds: float = 3.0,
    thickness: int = 1,
    color: Tuple[int, int, int] = (255, 255, 0),
    fps: float = 60.0
) -> np.ndarray:
    """
    Overlay a time-fading trajectory onto a BGR frame.

    A single O(log N) binary search finds the first point still inside the
    fading window; we then draw at most `window` points (≤ n_seconds·fps),
    so per-frame complexity is bounded and independent of the full list size.
    """
    window = int(round(n_seconds * fps))
    if window <= 0 or len(trajectory) < 2:
        return frame

    t_min = current_t - window
    # --- binary-search for first point with t_idx >= t_min ------------------
    # build a helper list of just the time stamps once; cheap even if repeated
    # but you can keep it outside if you manage the structure yourself
    time_stamps = [p[0] for p in trajectory]
    start_idx = bisect.bisect_left(time_stamps, t_min)

    # nothing inside window?
    if start_idx >= len(trajectory) - 1:
        return frame

    pts_slice = trajectory[start_idx:]          # <= window points
    out = frame.copy()

    for i in range(1, len(pts_slice)):
        t1, x1, y1 = pts_slice[i - 1]
        t2, x2, y2 = pts_slice[i]

        # use the *newer* point's age for the segment opacity
        age = current_t - t2
        if age > window or age < 0:
            continue

        alpha = 1.0 - (age / window)            # linear fade 1 → 0
        overlay = np.zeros_like(frame, np.uint8)
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, out, 1.0, 0.0, dst=out)

    return out