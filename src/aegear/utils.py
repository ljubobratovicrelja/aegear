from pathlib import Path

import sys
import os
import re

from datetime import datetime

import numpy as np


def resource_path(relative_path: str) -> Path:
    """Get the absolute path to the resource, works for dev and PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        # Go two levels up from aegear/app.py → project root
        base_path = Path(__file__).resolve().parents[2]
    return base_path / relative_path


def get_latest_model_path(directory, model_name):
    """
    Find the latest model file in the given directory matching the base model name.
    Model files are expected to be named as: modelname_YYYY-MM-DD.pth
    """
    pattern = re.compile(rf"{re.escape(model_name)}_(\d{{4}}-\d{{2}}-\d{{2}})\.pth")
    latest_date = None
    latest_file = None

    for filename in os.listdir(directory):
        match = pattern.fullmatch(filename)
        if match:
            date_str = match.group(1)
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if latest_date is None or file_date > latest_date:
                    latest_date = file_date
                    latest_file = filename
            except ValueError:
                continue

    return os.path.join(directory, latest_file) if latest_file else None


class Kalman2D:
    """A simple 2D Kalman filter for tracking."""

    def __init__(self, r=1.0, q=0.1):
        """Initialize the Kalman filter.
        
        Parameters
        ----------
        r : float
            The measurement noise.
        q : float
            The process noise.
        """
        self.x = np.zeros((4, 1))  # state
        self.P = np.eye(4) * 1000  # uncertainty

        self.A = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        self.R = np.eye(2) * r # measurement noise
        self.Q = np.eye(4) * q # process noise

    def reset(self, x, y):
        self.x = np.array([[x], [y], [0], [0]])
        self.P = np.eye(4)

    def update(self, z):
        # Predict
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

        # Update
        z = np.array(z).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[0, 0], self.x[1, 0]
