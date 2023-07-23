"""
Maze calibration module.

This module is used to calibrate the maze. It is used to calculate the pixel to cm ratio and the perspective transform.

"""
import cv2
import numpy as np


class MazeCalibration:

    # the real points of the maze
    REAL_PTS = np.array([[0, 0], [149.0, 5.0], [149.0, 35.0], [0.0, 40.0]], dtype=np.float32)

    def __init__(self, calibration_path):
        """
        Constructor.
        
        Parameters
        ----------
        calibration_path : str
            Path to the calibration file.
        """
        self.mtx, self.dist = self._load_calibration(calibration_path)
        self._perspectiveTransform = None

    def _load_calibration(self, calibration_path):
        storage = cv2.FileStorage(calibration_path, cv2.FILE_STORAGE_READ)
        mtx = storage.getNode("mtx").mat()
        dist = storage.getNode("dist").mat()
        storage.release()

        return (mtx, dist)

    
    def calibrate(self, screen_pts):
        """
        Run the maze characterization.
        
        Parameters
        ----------
        screen_pts : list
            The screen points to use for calibration.
        
        Returns
        -------
        float
            The pixel to cm ratio.

        """

        # -------------------
        # calcualte pixel to cm ratio and perspective transform
        # -------------------

        sample_pts = np.array(screen_pts, dtype=np.float32)

        img_scaling_factor = (sample_pts[1, 0] - sample_pts[0, 0]) / (
            MazeCalibration.REAL_PTS[1, 0] - MazeCalibration.REAL_PTS[0, 0]
        )

        # move points to match starting x position of samples, and scale up to image scale
        transformed_real_pts = MazeCalibration.REAL_PTS * img_scaling_factor + sample_pts[0, :]

        # do perspective transform to rectify image
        persp_T = cv2.getPerspectiveTransform(sample_pts, transformed_real_pts)

        # add homogeneous coordinate
        sample_pts = np.hstack((sample_pts, np.ones((4, 1))))

        # also warp points to be able to calculate pixel to cm ratio
        sample_pts = np.dot(persp_T, sample_pts.T).T

        # divide by homogeneous coordinate
        sample_pts = sample_pts[:, 0:2] / sample_pts[:, 2].reshape((4, 1))

        # now calculate pixel to cm ratio
        pixel_to_cm_ratio = np.linalg.norm(MazeCalibration.REAL_PTS[1, :] - MazeCalibration.REAL_PTS[0, :]) / np.linalg.norm(
            sample_pts[1, :] - sample_pts[0, :]
        )

        self._perspectiveTransform = persp_T

        return pixel_to_cm_ratio
    
    def rectify_image(self, image):
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