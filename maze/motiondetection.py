"""
Motion detection module for the maze project.
"""

import cv2
import numpy as np


class MotionDetector:
    """
    Motion detector class for the maze project.
    """

    MIN_AREA = 10

    def __init__(self, motion_threshold, erode_kernel_size=3, dilate_kernel_size=15, min_area=800, max_area=3000):
        self.motion_threshold = motion_threshold
        self.erode_kernel_size = erode_kernel_size
        self.dilate_kernel_size = dilate_kernel_size
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, prev_frame, this_frame, next_frame):
        # turn to grayspace
        gprev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gframe = cv2.cvtColor(this_frame, cv2.COLOR_BGR2GRAY)
        gnext_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        # calculate distance
        dist_prev = (np.abs(gframe.astype(np.float32) - gprev_frame.astype(np.float32))).astype(np.uint8)
        dist_next = (np.abs(gframe.astype(np.float32) - gnext_frame.astype(np.float32))).astype(np.uint8)

        _, dst_prev = cv2.threshold(dist_prev, self.motion_threshold, 255, cv2.THRESH_BINARY)
        _, dst_next = cv2.threshold(dist_next, self.motion_threshold, 255, cv2.THRESH_BINARY)

        # combine distances
        dst = cv2.bitwise_or(dst_prev, dst_next)

        # morphological closing
        dst = cv2.erode(dst, cv2.getStructuringElement(cv2.MORPH_RECT, (self.erode_kernel_size, self.erode_kernel_size)))
        dst = cv2.dilate(dst, cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilate_kernel_size, self.dilate_kernel_size)))

        dst = cv2.GaussianBlur(dst, (19, 19), 5.0)
        _, dst = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        good_contours, bad_contours = [], []

        for c in contours:
            area = cv2.contourArea(c)

            if area < MotionDetector.MIN_AREA:
                continue
            
            (bad_contours if area < self.min_area or area > self.max_area else good_contours).append(c)

        return good_contours, bad_contours