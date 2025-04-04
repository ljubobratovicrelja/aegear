"""
Scene calibration module.

This module is used to calibrate the camera and the scene size to get the pixel to cm ratio.
It includes a class `SceneCalibration` that handles the calibration process, including loading camera parameters,
assigning scene reference points, calibrating the scene, and rectifying images.
The calibration is performed using a set of screen points and a set of real-world reference points.

The class also provides a method to rectify images based on the calibration parameters.
It uses OpenCV for image processing and assumes that the camera calibration parameters are stored in a file.
The calibration points are expected to be in a specific order: top left, top right, bottom right, bottom left.

Note that this reference matching system is put in place due allow inconsistent camera placement with respect
to the original take of the calibration pattern. This calibration system uses this information to rectify the image
for easier tracking of the fish, and to estimate the pixel to cm ratio, hence allowing the correct metric tracking
of the fish within the experiment.
"""
from typing import List, Tuple

import cv2
import numpy as np


class SceneCalibration:
    """
    Calibration of the camera and the scene size to get the pixel to cm ratio.
    """

    # Sample points used in the Russian Sturgeon experiment, Fazekas et al, 2025.
    DEFAULT_SCENE_REF = np.array([[0, 0], [149.0, 5.0], [149.0, 35.0], [0.0, 40.0]], dtype=np.float32)

    def __init__(self, calibration_path: str, scene_reference=DEFAULT_SCENE_REF):
        """
        Constructor.
        
        Parameters
        ----------
        calibration_path : str
            Path to the calibration file.
        scene_reference : np.ndarray, optional
            The reference points for the scene. 4x2 array of floats, designating the borders
            of the reference area used for final image rectification and pixel to cm ratio calculation.
            The default value is assume from the Russian Sturgeon experiment, Fazekas et al, 2025.
        """
        self.mtx, self.dist = self._load_calibration(calibration_path)
        self._scene_reference = scene_reference
        self._perspectiveTransform = None

    def _load_calibration(self, calibration_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the camera calibration parameters from a file.
        """

        storage = cv2.FileStorage(calibration_path, cv2.FILE_STORAGE_READ)
        mtx = storage.getNode("mtx").mat()
        dist = storage.getNode("dist").mat()
        storage.release()

        return (mtx, dist)

    def assign_scene_calibration(self, points: List[Tuple[float, float]]):
        """
        Assign the scene calibration points.
        
        Parameters
        ----------

        points : list
            The scene reference points to use for calibration.
            The 4x2 array of floats, designating the borders of the reference area used for final image rectification and pixel to cm ratio calculation.
            By convention, the points are in the order: top left, top right, bottom right, bottom left.
        """
        points = np.array(points, dtype=np.float32)
        assert points.shape == (4, 2), "Real points must be a 4x2 array"
        self._scene_reference = points
    
    def calibrate(self, screen_pts: List[Tuple[float, float]]) -> float:
        """
        Run the maze characterization.
        
        Parameters
        ----------
        screen_pts : list
            The screen points to use for calibration, which within the scene match the points assigned for the scene reference.
            As for the reference points, the points are in the order: top left, top right, bottom right, bottom left.
        
        Returns
        -------
        float
            The pixel to cm ratio.
        """
        sample_pts = np.array(screen_pts, dtype=np.float32)
        assert sample_pts.shape == (4, 2), "Screen points must be a 4x2 array"

        img_scaling_factor = (sample_pts[1, 0] - sample_pts[0, 0]) / (
            self._scene_reference[1, 0] - self._scene_reference[0, 0]
        )

        # move points to match starting x position of samples, and scale up to image scale
        transformed_real_pts = self._scene_reference * img_scaling_factor + sample_pts[0, :]

        # do perspective transform to rectify image
        persp_T = cv2.getPerspectiveTransform(sample_pts, transformed_real_pts)

        # add homogeneous coordinate
        sample_pts = np.hstack((sample_pts, np.ones((4, 1))))

        # also warp points to be able to calculate pixel to cm ratio
        sample_pts = np.dot(persp_T, sample_pts.T).T

        # divide by homogeneous coordinate
        sample_pts = sample_pts[:, 0:2] / sample_pts[:, 2].reshape((4, 1))

        # now calculate pixel to cm ratio
        pixel_to_cm_ratio = np.linalg.norm(self._scene_reference[1, :] - self._scene_reference[0, :]) / np.linalg.norm(
            sample_pts[1, :] - sample_pts[0, :]
        )

        self._perspectiveTransform = persp_T

        return pixel_to_cm_ratio
    
    def rectify_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rectify the image.
        
        Parameters
        ----------
        image : np.ndarray
            The image to rectify.
        
        Returns
        -------
        np.ndarray
            The rectified image.
        
        """
        assert self._perspectiveTransform is not None, "Need to calibrate first"

        ret_image = cv2.undistort(image, self.mtx, self.dist)
        ret_image = cv2.warpPerspective(ret_image, self._perspectiveTransform, image.shape[0:2][::-1])

        return ret_image