import os
import glob
import json
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from scipy.signal import savgol_filter

from PIL import Image
from moviepy.video.io.VideoFileClip import VideoFileClip

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


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


class FishHeatmapSequenceDataset(Dataset):
    def __init__(
        self,
        tracking_json_path,
        unet_model,
        seq_len=5,
        max_span=50,
        min_confidence=0.5,
        crop_size=129,
        transform=None,
        device="cuda"
    ):
        """
        Args:
            tracking_json_path (str): Path to the JSON with tracking data.
            unet_model (nn.Module): Trained EfficientUNet model.
            seq_len (int): Number of consecutive tracking points to use.
            max_span (int): Max frame span allowed within a sequence.
            min_confidence (float): Minimum confidence to use a tracking point.
            crop_size (int): Size of the square ROI to extract around centroid.
            transform: Torchvision transform to apply to crops before U-Net.
            device (str): Device to place input tensors and model.
        """
        with open(tracking_json_path, 'r') as f:
            data = json.load(f)

        self.video_path = data["video"]
        self.tracking = [
            t for t in data["tracking"]
            if t["confidence"] >= min_confidence
        ]

        self.seq_len = seq_len
        self.max_span = max_span
        self.crop_size = crop_size
        self.transform = transform or transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()
        ])
        self.unet = unet_model.to(device).eval()
        self.device = device

        self.valid_sequences = self._build_valid_sequences()
        self.video = VideoFileClip(self.video_path)

    def _build_valid_sequences(self):
        sequences = []
        for i in range(len(self.tracking) - self.seq_len + 1):
            group = self.tracking[i:i + self.seq_len]
            frame_ids = [pt["frame_id"] for pt in group]

            if max(frame_ids) - min(frame_ids) > self.max_span:
                continue

            coords = np.array([pt["coordinates"] for pt in group])
            frames = np.array(frame_ids)

            sort_idx = np.argsort(frames)
            frames_sorted = frames[sort_idx]
            coords_sorted = coords[sort_idx]

            if len(coords_sorted) >= 5:
                window = min(7, len(coords_sorted) // 2 * 2 + 1)
                try:
                    x_smooth = savgol_filter(coords_sorted[:, 0], window_length=window, polyorder=2)
                    y_smooth = savgol_filter(coords_sorted[:, 1], window_length=window, polyorder=2)
                except ValueError:
                    continue

                smoothed = list(zip(x_smooth, y_smooth))
                sequences.append({
                    "frames": frames_sorted.tolist(),
                    "coordinates": smoothed
                })

        return sequences

    def __len__(self):
        return len(self.valid_sequences)

    def _load_frame(self, t):
        frame = self.video.get_frame(t / self.video.fps)  # RGB
        return Image.fromarray((frame * 255).astype(np.uint8))

    def _crop_roi(self, img, center, size):
        x, y = int(center[0]), int(center[1])
        left = max(0, x - size // 2)
        top = max(0, y - size // 2)
        img = TF.crop(img, top, left, size, size)
        return img

    def __getitem__(self, idx):
        seq = self.valid_sequences[idx]
        frames = seq["frames"]
        coords = seq["coordinates"]

        crops = []
        for t, center in zip(frames, coords):
            img = self._load_frame(t)
            crop = self._crop_roi(img, center, self.crop_size)
            crop = self.transform(crop).unsqueeze(0).to(self.device)  # [1, C, 224, 224]

            with torch.no_grad():
                heatmap = torch.sigmoid(self.unet(crop))  # [1, 1, 224, 224]

            crops.append(heatmap.squeeze(0).cpu())  # [1, 224, 224]

        heatmap_seq = torch.stack(crops)  # [T, 1, 224, 224]
        target = torch.tensor(coords[-1], dtype=torch.float32)  # Predict final centroid

        return heatmap_seq, target