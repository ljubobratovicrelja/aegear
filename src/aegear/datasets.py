import os
import time
import glob
import json
import random
import pickle
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from scipy.interpolate import Rbf

from PIL import Image

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import torchvision.transforms as transforms


class FishHeatmapDataset(Dataset):
    def __init__(self, annotation_data, image_dir, heatmap_dir,
                 background_dir=None, background_prob=0.3,
                 joint_transform=None, augmentation_transform=None,
                 exclude_indices=None):
        self.samples = []
        self.joint_transform = joint_transform
        self.augmentation_transform = augmentation_transform
        self.exclude_indices = set(exclude_indices or [])

        for img_info in annotation_data['images']:
            file_name = img_info['file_name']
            img_path = os.path.join(image_dir, file_name)
            heatmap_path = os.path.join(heatmap_dir, os.path.splitext(file_name)[0] + '.npy')

            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            heatmap = np.load(heatmap_path)

            self.samples.append((image, heatmap))

        # Exclude flagged samples
        self.samples = [s for i, s in enumerate(self.samples) if i not in self.exclude_indices]
        print(f"Loaded {len(self.samples)} samples after excluding {len(self.exclude_indices)} flagged samples.")

        # Add background samples
        if background_dir:
            # Do glob for PNG files in the background directory
            background_files = glob.glob(os.path.join(background_dir, '*.png'))
            random.shuffle(background_files)

            num_background_samples = min(len(background_files), int(len(self.samples) * background_prob))
            print(f"Adding {num_background_samples} background samples from {len(background_files)} available files.")

            # Add samples to the dataset
            for i in range(num_background_samples):
                img = cv2.imread(background_files[i])

                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                self.samples.append((img, heatmap))
        
        # Shuffle the dataset
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, heatmap = self.samples[idx]

        # Convert image and heatmap to PIL
        image = Image.fromarray(image)
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))  # 0–255 grayscale

        # Apply joint transforms (same seed for both)
        if self.joint_transform:
            seed = np.random.randint(0, 10000)
            torch.manual_seed(seed)
            image = self.joint_transform(image)
            torch.manual_seed(seed)
            heatmap_img = self.joint_transform(heatmap_img)

        # Turn image to tensor and normalize to [0,1]
        image = transforms.ToTensor()(image).clamp(0, 1).float()
        
        if self.augmentation_transform:
            image = self.augmentation_transform(image.unsqueeze(0)).squeeze(0)

        # Standardize the image.
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])(image)

        # Convert heatmap to tensor and normalize to [0,1]
        heatmap_tensor = transforms.ToTensor()(heatmap_img).clamp(0, 1).float()  # shape [1, H, W]

        return image, heatmap_tensor

def split_coco_annotations(
    coco_json_path: Path,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[dict, dict]:
    """
    Loads a COCO JSON and splits it into train/val dictionaries based on image-level split.
    
    Args:
        coco_json_path (Path): Path to the COCO annotations.json.
        train_ratio (float): Ratio of images to assign to the training set.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[dict, dict]: (train_dict, val_dict)
    """
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # Reproducible shuffle
    random.seed(seed)
    shuffled_images = images[:]
    random.shuffle(shuffled_images)

    split_idx = int(len(shuffled_images) * train_ratio)
    train_images = shuffled_images[:split_idx]
    val_images = shuffled_images[split_idx:]

    train_img_ids = {img["id"] for img in train_images}
    val_img_ids = {img["id"] for img in val_images}

    # Filter annotations
    train_annotations = [ann for ann in annotations if ann["image_id"] in train_img_ids]
    val_annotations = [ann for ann in annotations if ann["image_id"] in val_img_ids]

    train_dict = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": categories
    }

    val_dict = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": categories
    }

    return train_dict, val_dict

class RandomPoissonNoise(torch.nn.Module):
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x

        x_scaled = x * 255.0
        noise = torch.poisson(x_scaled)
        return torch.clamp(noise / 255.0, 0.0, 1.0)


class TrackingDataset(Dataset):

    _CROP_SIZE = 129 # Size of the square ROI to extract around centroid
    _OUTPUT_SIZE = 28 # Size of the heatmap output

    def __init__(
        self,
        tracking_json_path,
        video_dir="",
        future_frame_seek=[1, 3, 5, 7],
        interpolation_smoothness=0.5,
        max_epoch_samples=0,
        randomized=False,
        augmentation_transform=None,
    ):
        with open(tracking_json_path, 'r') as f:
            data = json.load(f)

        self.video_path = os.path.join(video_dir, data["video"])
        self.tracking = data["tracking"]
        self.smooth_trajectory, self.min_frame, self.max_frame = self._interpolate_tracking(interpolation_smoothness)
        self.future_frame_seek = future_frame_seek
        self.max_epoch_samples = max_epoch_samples
        self.randomized = randomized

        # Estimate FPS from video file
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            raise Exception(f"Could not open video file: {self.video_path}")

        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = np.array([self.frame_width, self.frame_height])

        self.tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])

        self.augmentation_transform = augmentation_transform

        self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )


    def _interpolate_tracking(self, interpolation_smoothness):
        frame_ids = np.array([pt["frame_id"] for pt in self.tracking])
        coords = np.array([pt["coordinates"] for pt in self.tracking])

        min_frame = int(frame_ids.min())
        max_frame = int(frame_ids.max())
        dense_frames = np.arange(min_frame, max_frame + 1)

        rbf_x = Rbf(frame_ids, coords[:, 0], function='multiquadric', epsilon=interpolation_smoothness)
        rbf_y = Rbf(frame_ids, coords[:, 1], function='multiquadric', epsilon=interpolation_smoothness)

        x_interp = rbf_x(dense_frames)
        y_interp = rbf_y(dense_frames)

        trajectory = np.stack([x_interp, y_interp], axis=1)

        return trajectory, min_frame, max_frame
    
    def test_sequence_cache(self):
        for frame_id in range(self.min_frame, self.max_frame):
            try:
                frame = self._read_frame(frame_id)
            except:
                print(f"Frame {frame_id} not found in video {self.video_path}")
                continue

            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            coodinate = self.smooth_trajectory[frame_id - self.min_frame]

            cv2.circle(img, (int(coodinate[0]), int(coodinate[1])), 5, (0, 255, 0), -1)

            cv2.imshow("Test", np.array(img))
            cv2.waitKey(0)

    def _read_frame(self, frame_id):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, img = self.video.read()
        if not ret:
            raise Exception(f"Could not read frame {frame_id} from video {self.video_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _get_crop(self, frame_id, center):
        frame = self._read_frame(frame_id)

        x1 = int(center[0] - self._CROP_SIZE // 2)
        y1 = int(center[1] - self._CROP_SIZE // 2)
        x2 = x1 + self._CROP_SIZE
        y2 = y1 + self._CROP_SIZE

        # Bail out if crop goes out of bounds
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            raise IndexError("Crop out of bounds")

        return frame[y1:y2, x1:x2, :]

    @staticmethod
    def generate_gaussian_heatmap(center, sigma=2.0):
        x = torch.arange(0, TrackingDataset._OUTPUT_SIZE, 1).float()
        y = torch.arange(0, TrackingDataset._OUTPUT_SIZE, 1).float()
        y = y[:, None]

        x0, y0 = center
        heatmap = torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        return heatmap
    
    def num_samples(self):
        return self.max_frame - self.min_frame - max(self.future_frame_seek) - 1

    def __len__(self):
        if self.max_epoch_samples > 0:
            return min(self.max_epoch_samples, self.num_samples())
        return self.num_samples()

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def __getitem__(self, idx):
        if self.randomized and self.max_epoch_samples > 0:
            idx = random.choice(range(self.num_samples()))

        # Reset seed with  time for max randomness
        frame_jump = random.choice(self.future_frame_seek)

        template_frame_id = self.min_frame + idx
        search_frame_id = template_frame_id + frame_jump

        template_coordinate = self.smooth_trajectory[idx]
        search_coordinate = self.smooth_trajectory[idx + frame_jump]

        try:
            template = self._get_crop(template_frame_id, template_coordinate)
            search = self._get_crop(search_frame_id, template_coordinate)
        except IndexError:
            return self.__getitem__((idx + 1) % len(self))
        
        template = self.tensor_transforms(template)
        search = self.tensor_transforms(search)

        # Augmentation with same seed
        if self.augmentation_transform:
            seed = np.random.randint(0, 10000)
            torch.manual_seed(seed)
            template = self.augmentation_transform(template.unsqueeze(0)).squeeze(0)
            torch.manual_seed(seed)
            search = self.augmentation_transform(search.unsqueeze(0)).squeeze(0)

        # Normalize the images
        template = self.normalize(template)
        search = self.normalize(search)

        heatmap_scale_diff = self._OUTPUT_SIZE / self._CROP_SIZE
        search_roi_hit = (search_coordinate - template_coordinate) * heatmap_scale_diff + self._OUTPUT_SIZE // 2
        heatmap = TrackingDataset.generate_gaussian_heatmap(search_roi_hit, sigma=2.0).unsqueeze(0)

        return (
            template, search, heatmap
        )
        