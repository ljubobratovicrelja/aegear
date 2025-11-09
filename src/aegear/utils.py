from pathlib import Path

import sys
import os
import re

import torch
import zipfile
from google.cloud import storage

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


def get_latest_model_path(directory, model_name) -> str:
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


def load_model_with_weights(model_class, model_path, device):
    """Load a model with weights from a checkpoint.
    
    Parameters
    ----------
    model_class : torch.nn.Module
        Model class to instantiate
    model_path : str
        Path to model checkpoint
    device : str
        Device to load model on ('cuda' or 'cpu')
        
    Returns
    -------
    torch.nn.Module
        Loaded model
    """
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def download_dataset(dataset_dir, dataset_type="tracking"):
    """Download dataset from GCS if not already present.
    
    Parameters
    ----------
    dataset_dir : str
        Directory to download the dataset to
    dataset_type : str
        Type of dataset to download ('tracking' or 'detection')
    """
    bucket_name = "aegear-training-data"
    blob_path = f"cache/{dataset_type}.zip"
    
    dataset_path = os.path.join(dataset_dir, dataset_type)
    
    if os.path.exists(dataset_path):
        print(f"{dataset_type.capitalize()} dataset already exists. Skipping download.")
        return
    
    print(f"{dataset_type.capitalize()} dataset not found. Downloading...")
    zip_path = os.path.join(dataset_dir, f"{dataset_type}.zip")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download the zip file if it doesn't exist
    if not os.path.exists(zip_path):
        print(f"Downloading gs://{bucket_name}/{blob_path} to {zip_path}...")
        
        # Initialize anonymous GCS client for public data
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(zip_path)
        
        print("Download complete.")
    
    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
        print(f"Extracted to {dataset_dir}")


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
