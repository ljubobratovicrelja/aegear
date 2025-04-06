import os
import glob
import json
import random
import pickle
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import Rbf

from PIL import Image

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from tqdm import tqdm

from aegear.model import EfficientUNet


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
        unet_model: EfficientUNet,
        device=None,
        history_len=30,
        future_len=5,
        stride=5,
        max_span=50,
        interpolation_smoothness=0.5,
        min_confidence=0.5,
        crop_size=129,
        cache_path=None,
    ):
        """
        Args:
            tracking_json_path (str): Path to the JSON with tracking data.
            unet_model (nn.Module): Trained EfficientUNet model.
            history_len (int): Number of historical steps (excluding the target) for interpolation.
            future_len (int): Number of future steps to predict.
            stride (int): Step size between frames in the sequence.
            max_span (int): Max frame span allowed within a sequence.
            min_confidence (float): Minimum confidence to use a tracking point.
            crop_size (int): Size of the square ROI to extract around centroid.
            cache_path (str): Path to cache ROIs and heatmaps.
        """
        with open(tracking_json_path, 'r') as f:
            data = json.load(f)

        self.video_path = data["video"]
        self.tracking = [
            t for t in data["tracking"]
            if t["confidence"] >= min_confidence
        ]

        self.interpolation_smoothness = interpolation_smoothness
        self.history_len = history_len
        self.future_len = future_len
        self.stride = stride
        self.max_span = max_span
        self.crop_size = crop_size

        unet_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        unet_model.eval()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unet_model.to(device)

        self.sequence_cache = self._build_valid_sequence(
            unet_model, unet_transforms, device, cache_path=cache_path
        )
    
    def _interpolate_tracking(self):
        frame_ids = np.array([pt["frame_id"] for pt in self.tracking])
        coords = np.array([pt["coordinates"] for pt in self.tracking])

        min_frame = int(frame_ids.min())
        max_frame = int(frame_ids.max())
        dense_frames = np.arange(min_frame, max_frame + 1)

        rbf_x = Rbf(frame_ids, coords[:, 0], function='multiquadric', epsilon=self.interpolation_smoothness)
        rbf_y = Rbf(frame_ids, coords[:, 1], function='multiquadric', epsilon=self.interpolation_smoothness)

        x_interp = rbf_x(dense_frames)
        y_interp = rbf_y(dense_frames)

        # Cache the dense interpolation: maps frame_id -> (x, y)
        self._cached_trajectory_range = (min_frame, max_frame)
        self._cached_trajectory = np.stack([x_interp, y_interp], axis=1)
    
    def _get_interpolated_trajectory(self, frame_range):
        if not hasattr(self, "_cached_trajectory"):
            self._interpolate_tracking()

        min_frame, _ = self._cached_trajectory_range
        indices = np.array(frame_range) - min_frame

        return self._cached_trajectory[indices]
    
    def _build_valid_sequence(self, model, transforms, device, cache_path=None):
        begin_frame = self.tracking[0]["frame_id"]
        end_frame = self.tracking[-1]["frame_id"]

        sequence_cache = []
        sequence_path = os.path.join(cache_path, f"sequence_cache.pkl")

        if cache_path and os.path.exists(sequence_path):
            with open(sequence_path, 'rb') as f:
                sequence_cache = pickle.load(f)
            
            if sequence_cache:
                print(f"Loaded cached sequence data from {cache_path}")
                return sequence_cache

        print(f"Building sequence cache for frames {begin_frame} to {end_frame}...")

        valid_frame_range = range(begin_frame, end_frame + 1)
        print(f"Computing interpolated trajectory for frames {begin_frame} to {end_frame}...")
        valid_coords = self._get_interpolated_trajectory(valid_frame_range)
        print("Interpolated trajectory computed.")

        cap = cv2.VideoCapture(self.video_path)

        data_bar = tqdm(zip(range(begin_frame, end_frame + 1), valid_coords), "Processing frames...")
        for fid, coord in data_bar:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()

            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            roi = self._crop_roi(img, coord, self.crop_size)

            h_roi = transforms(roi).unsqueeze(0).to(device)
            with torch.no_grad():
                heatmap = torch.sigmoid(model(h_roi)).squeeze(0).cpu()

            sequence_cache.append({
                "frame_id": fid,
                "coordinates": coord,
                "roi": roi,
                "heatmap": heatmap
            })

        cap.release()

        # Store the cache
        if cache_path:
            os.makedirs(cache_path, exist_ok=True)

            with open(sequence_path, 'wb') as f:
                pickle.dump(sequence_cache, f)

            print(f"Cached sequence data to {sequence_path}")

        return sequence_cache

    def __len__(self):
        return len(self.sequence_cache) - self.history_len - self.future_len - 1

    def _crop_roi(self, img, center, size):
        x, y = int(center[0]), int(center[1])
        left = max(0, x - size // 2)
        top = max(0, y - size // 2)
        img = TF.crop(img, top, left, size, size)
        return img

    def __getitem__(self, idx):
        
        present_frame = idx + self.history_len

        history = self.sequence_cache[idx:present_frame:self.stride]
        present = [self.sequence_cache[present_frame]]
        future = self.sequence_cache[present_frame + self.stride:present_frame + self.stride + self.future_len:self.stride]

        coordinates = [s["coordinates"] for s in (history + present + future)]
        heatmaps = [h["heatmap"] for h in (history + present)]

        heatmap_seq = torch.stack(heatmaps)  # [T, 1, H, W], where T = seq_len - 1

        # Reference for the relative offset computation
        ref_x, ref_y = coordinates[-(len(future) + 1)]

        # Compute relative (dx, dy) offsets for each input frame w.r.t. f₀
        relative_offsets = [
            [x - ref_x, y - ref_y]
            for (x, y) in coordinates
        ]

        present_point = len(history) + 1

        past_t = torch.tensor(relative_offsets[0:present_point], dtype=torch.float32)  # [2]
        future_t = torch.tensor(relative_offsets[present_point:], dtype=torch.float32)  # [2]
        
        return heatmap_seq, past_t, future_t