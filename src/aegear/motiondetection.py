"""
Motion detection module.

This module provides the MotionDetector class that identifies motion by comparing
three consecutive frames. The algorithm converts frames to grayscale, computes the
absolute difference between frames, applies binary thresholding, combines the results,
and uses morphological operations to filter the motion regions before extracting contours.
"""

import cv2
import numpy as np
from typing import Tuple, List


class MotionDetector:
    """
    Motion detector class that identifies motion by comparing three consecutive frames.
    """

    MIN_AREA: int = 10

    def __init__(self, motion_threshold: int, erode_kernel_size: int = 3,
                 dilate_kernel_size: int = 15, min_area: int = 800, max_area: int = 3000) -> None:
        """
        Initialize the MotionDetector.

        Parameters
        ----------
        motion_threshold : int
            The threshold used to detect motion based on pixel intensity difference.
        erode_kernel_size : int, optional
            Size of the kernel used for erosion (default is 3).
        dilate_kernel_size : int, optional
            Size of the kernel used for dilation (default is 15).
        min_area : int, optional
            Minimum contour area to be considered as good motion (default is 800).
        max_area : int, optional
            Maximum contour area to be considered as good motion (default is 3000).
        """
        self.motion_threshold = motion_threshold
        self.erode_kernel_size = erode_kernel_size
        self.dilate_kernel_size = dilate_kernel_size
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, prev_frame: np.ndarray, this_frame: np.ndarray,
               next_frame: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Detect motion by comparing three consecutive frames.

        The function converts the frames to grayscale, computes the absolute differences,
        thresholds them to produce binary images, combines the thresholded images, applies
        morphological operations to remove noise, and finally extracts contours. Detected
        contours are classified into "good" (within the area range) and "bad" (outside the
        area range but above a minimum threshold).

        Parameters
        ----------
        prev_frame : numpy.ndarray
            Previous frame in BGR color space.
        this_frame : numpy.ndarray
            Current frame in BGR color space.
        next_frame : numpy.ndarray
            Next frame in BGR color space.

        Returns
        -------
        Tuple[List[numpy.ndarray], List[numpy.ndarray]]
            A tuple containing two lists of contours:
            - The first list contains contours with areas between min_area and max_area.
            - The second list contains contours with areas outside that range but above MIN_AREA.
        """
        # Convert frames to grayscale
        gprev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gframe = cv2.cvtColor(this_frame, cv2.COLOR_BGR2GRAY)
        gnext_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # Compute absolute differences between the current frame and its neighbors
        diff_prev = np.abs(gframe.astype(np.float32) - gprev_frame.astype(np.float32)).astype(np.uint8)
        diff_next = np.abs(gframe.astype(np.float32) - gnext_frame.astype(np.float32)).astype(np.uint8)

        # Apply binary thresholding to highlight significant differences
        _, thresh_prev = cv2.threshold(diff_prev, self.motion_threshold, 255, cv2.THRESH_BINARY)
        _, thresh_next = cv2.threshold(diff_next, self.motion_threshold, 255, cv2.THRESH_BINARY)

        # Combine the thresholded images
        combined = cv2.bitwise_or(thresh_prev, thresh_next)

        # Apply morphological operations to reduce noise and close gaps
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.erode_kernel_size, self.erode_kernel_size))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilate_kernel_size, self.dilate_kernel_size))
        morphed = cv2.erode(combined, erode_kernel)
        morphed = cv2.dilate(morphed, dilate_kernel)

        # Smooth the image and reapply thresholding to finalize the binary image
        blurred = cv2.GaussianBlur(morphed, (19, 19), 5.0)
        _, final_thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(final_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        good_contours: List[np.ndarray] = []
        bad_contours: List[np.ndarray] = []

        # Classify contours based on their area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MotionDetector.MIN_AREA:
                continue

            if self.min_area <= area <= self.max_area:
                good_contours.append(contour)
            else:
                bad_contours.append(contour)

        return good_contours, bad_contours
