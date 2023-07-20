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
        self._pts = []
        self._image = None
        self._winName = "Select 4 points"
        self._perspectiveTransform = None

    def _load_calibration(self, calibration_path):
        storage = cv2.FileStorage(calibration_path, cv2.FILE_STORAGE_READ)
        mtx = storage.getNode("mtx").mat()
        dist = storage.getNode("dist").mat()
        storage.release()

        return (mtx, dist)

    def _select_points(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._pts.append((x, y))


    
    def calibrate(self, image):
        """
        Run the maze characterization.
        
        Parameters
        ----------
        image : np.ndarray
            The image to run the maze characterization on.
        
        Returns
        -------
        float
            The pixel to cm ratio.

        """
        self._image = image.copy()  # update the image and then run the point clicking

        # undistort the image
        self._image = cv2.undistort(self._image, self.mtx, self.dist)

        cv2.namedWindow(self._winName)
        cv2.setMouseCallback(self._winName, self._select_points)

        while(True):
            if len(self._pts) == 4 or cv2.waitKey(20) & 0xFF == 27:  # Exit if ESC key is pressed or we have 4 points
                break

            dimage = self._image.copy()
            for pt in self._pts:
                dimage = cv2.circle(dimage, pt, 5, (0, 255, 0))
            cv2.imshow(self._winName, dimage)
        
        assert len(self._pts) == 4, "Need 4 points to calculate pixel to cm ratio"

        # -------------------
        # calcualte pixel to cm ratio and perspective transform
        # -------------------

        sample_pts = np.array(self._pts, dtype=np.float32)

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

        cv2.destroyWindow(self._winName)

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