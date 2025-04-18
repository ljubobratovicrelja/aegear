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
import torchvision.transforms.functional as F

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

    _HEATMAP_CACHE = None  # Class-level shared cache for heatmaps
    _CROP_SIZE = 129 # Size of the square ROI to extract around centroid
    _TENSOR_CROP_SIZE = 224

    def __init__(
        self,
        tracking_json_path,
        unet_model: EfficientUNet,
        video_dir="",
        device=None,
        history_lookback_s=(0.5, 1.0),
        n_history_samples=5,
        future_horizon_s=1.0,
        n_history_caches=0,
        interpolation_smoothness=0.5,
        min_confidence=0.5,
        cache_path=None,
    ):
        """
        Args:
            tracking_json_path (str): Path to the JSON with tracking data.
            unet_model (nn.Module): Trained EfficientUNet model.
            history_lookback_s (float): Time in seconds to look back for history sampling.
            n_history_samples (int): Number of historical samples to randomly draw from history window.
            future_horizon_s (float): Time in seconds to look ahead for future prediction.
            min_confidence (float): Minimum confidence to use a tracking point.
            crop_size (int): Size of the square ROI to extract around centroid.
            cache_path (str): Path to cache ROIs and heatmaps.
        """
        with open(tracking_json_path, 'r') as f:
            data = json.load(f)

        self.video_path = os.path.join(video_dir, data["video"])
        self.tracking = [
            t for t in data["tracking"]
            if t["confidence"] >= min_confidence
        ]

        self.interpolation_smoothness = interpolation_smoothness
        self.history_lookback_s = history_lookback_s
        self.n_history_samples = n_history_samples
        self.future_horizon_s = future_horizon_s

        self.unet_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        unet_model.eval()
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        unet_model.to(device)

        # Estimate FPS from video file
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            raise Exception(f"Could not open video file: {self.video_path}")
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Build the sequence cache.
        self.sequence_cache = self._build_valid_sequence(
            unet_model, device, cache_path=cache_path
        )

        for entry in self.sequence_cache:
            entry["timestamp"] = entry["frame_id"] / self.fps
            x, y = entry["coordinates"]
            entry["normalized_coordinates"] = [x / self.frame_width, y / self.frame_height]
        
        # Store data into dedicated members of easier fetching in __getitem__
        self.timestamps = np.array([e["timestamp"] for e in self.sequence_cache])
        self.norm_coords = np.array([e["normalized_coordinates"] for e in self.sequence_cache])

        if FishHeatmapSequenceDataset._HEATMAP_CACHE is None:
            print("Creating shared heatmap cache...")
            heatmap_tensor = torch.stack([e["heatmap"].squeeze(0) for e in self.sequence_cache], dim=0)  # [N, H, W]
            FishHeatmapSequenceDataset._HEATMAP_CACHE = heatmap_tensor.share_memory_()  # ensure shared memory
            print("Shared heatmap cache created.")

        self.heatmaps = FishHeatmapSequenceDataset._HEATMAP_CACHE

        self._item_cache = self._build_item_cache(n_history_caches)

    def _build_item_cache(self, n_history_caches):
        if n_history_caches <= 0:
            return None

        item_cache = []
        print(f"Building item cache with {n_history_caches} samples...")
        for history_lookback_s in np.linspace(self.history_lookback_s[0], self.history_lookback_s[1], n_history_caches):
            for i in tqdm(self._range_for_lookback(history_lookback_s), desc=f"Building item cache for level {history_lookback_s:.2f}s"):
                item_cache.append(self._get_item(i, history_lookback_s))

        return item_cache

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
    
    def _build_valid_sequence(self, model, device, cache_path=None):
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

        data_bar = tqdm(zip(range(begin_frame, end_frame + 1), valid_coords), "Processing frames...")
        for fid, coord in data_bar:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = self.video.read()

            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            roi = self._crop_roi(img, coord, self._CROP_SIZE)

            h_roi = self.unet_transforms(roi).unsqueeze(0).to(device)
            with torch.no_grad():
                heatmap = torch.sigmoid(model(h_roi)).squeeze(0).cpu()

            roi = self._crop_roi(img, coord, self._CROP_SIZE * 2)

            sequence_cache.append({
                "frame_id": fid,
                "coordinates": coord,
                "roi": roi,
                "heatmap": heatmap
            })

        # Store the cache
        if cache_path:
            os.makedirs(cache_path, exist_ok=True)

            with open(sequence_path, 'wb') as f:
                pickle.dump(sequence_cache, f)

            print(f"Cached sequence data to {sequence_path}")

        return sequence_cache
    
    def _range_for_lookback(self, history_lookback_s):
        total_time_span = history_lookback_s + self.future_horizon_s
        max_required_frames = int(total_time_span * self.fps) + 1  # +1 for the present frame
    
        # We make sure that the relative computation of the last history item is possible
        max_stride = int((history_lookback_s / self.n_history_samples) * self.fps)

        return range(max_stride, len(self.sequence_cache) - max_required_frames)

    def __len__(self):
        if self._item_cache is None:
            # Compute number of usable samples based on max time span needed
            _, max_lookback = self.history_lookback_s
            return len(self._range_for_lookback(max_lookback))
        else:
            # Take a sample from the cache
            return len(self._item_cache)

    def _crop_roi(self, img, center, size):
        x, y = int(center[0]), int(center[1])
        left = max(0, x - size // 2)
        top = max(0, y - size // 2)
        img = F.crop(img, top, left, size, size)
        return img

    def _crop_tensor(self, tensor, center, size):
        # tensor: [1, H, W]
        pad = size // 2
        tensor_padded = F.pad(tensor, (pad, pad, pad, pad), fill=0, padding_mode='constant')

        x, y = int(center[0]) + pad, int(center[1]) + pad
        x1 = x - size // 2
        y1 = y - size // 2
        x2 = x1 + size
        y2 = y1 + size

        return tensor_padded[..., y1:y2, x1:x2]
    
    def _get_item(self, idx, history_lookback_s):
        # How many frames we're jumping for this lookback.
        frame_stride = int((history_lookback_s / self.n_history_samples) * self.fps)

        # Ensure we are not going out of bounds when computing relative offset for
        # the last history sample
        idx = idx + frame_stride

        present_idx = idx + int(history_lookback_s * self.fps)
        present_ts = self.timestamps[present_idx]

        history_start_ts = present_ts - history_lookback_s
        start_idx = np.searchsorted(self.timestamps, history_start_ts, side='left')
        history_indices = np.arange(start_idx, present_idx)

        if len(history_indices) >= self.n_history_samples:
            sampled_idx = np.arange(present_idx - frame_stride * self.n_history_samples, present_idx, frame_stride)
            sampled_idx = sampled_idx[sampled_idx >= 0]

            while len(sampled_idx) < self.n_history_samples:
                sampled_idx = np.insert(sampled_idx, 0, sampled_idx[0])
        else:
            sampled_idx = history_indices

        # --- Collect history ---
        heatmap_seq = []
        history_ts = []
        for i, hist_idx in enumerate(sampled_idx):
            # center based on previous index (or itself if first)
            prev_idx = sampled_idx[i] - frame_stride

            present_coord = self.norm_coords[hist_idx]
            prev_coord = self.norm_coords[prev_idx]
            relative_offset = prev_coord - present_coord

            center_px = (
                relative_offset[0] * self.frame_width + self._TENSOR_CROP_SIZE // 2,
                relative_offset[1] * self.frame_height + self._TENSOR_CROP_SIZE // 2
            )

            full_heatmap = self.heatmaps[hist_idx][None, :, :]  # [1, H, W]
            cropped_heatmap = self._crop_tensor(full_heatmap, center_px, self._TENSOR_CROP_SIZE)  # [1, H, W]
            heatmap_seq.append(cropped_heatmap)
            history_ts.append(self.timestamps[hist_idx])

        heatmap_seq = torch.stack(heatmap_seq, dim=0)  # [T, 1, H, W]
        history_ts = torch.tensor(history_ts, dtype=torch.float32)  # [T]

        # --- Current RGB centered on last known anchor ---
        present_coord = self.norm_coords[present_idx]
        prev_coord = self.norm_coords[sampled_idx[-1]]
        relative_offset = prev_coord  - present_coord 

        # Shift center by this relative offset (measured in image pixels)
        center_px = (
            relative_offset[0] * self.frame_width + self._CROP_SIZE,
            relative_offset[1] * self.frame_height + self._CROP_SIZE 
        )

        full_rgb = self.sequence_cache[present_idx]["roi"]
        rgb_crop = self._crop_roi(full_rgb, center_px, self._CROP_SIZE)
        rgb_tensor = self.unet_transforms(rgb_crop)

        heatmap_scaling = self._TENSOR_CROP_SIZE / self._CROP_SIZE

        # --- Target heatmap ---
        target_full = self.heatmaps[present_idx][None, :, :]
        target = self._crop_tensor(target_full, (
            relative_offset[0] * heatmap_scaling * self.frame_width + self._TENSOR_CROP_SIZE // 2,
            relative_offset[1] * heatmap_scaling * self.frame_height + self._TENSOR_CROP_SIZE // 2,
        ), self._TENSOR_CROP_SIZE)

        return {
            "history_heatmaps": heatmap_seq,
            "history_timestamps": history_ts,
            "current_rgb": rgb_tensor,
            "current_timestamp": torch.tensor(present_ts, dtype=torch.float32),
            "target_heatmap": target
        }

    def __getitem__(self, idx):
        if self._item_cache is None:
            min_lookback_s, max_lookback_s = self.history_lookback_s
            return self._get_item(idx, np.random.uniform(min_lookback_s, max_lookback_s))
        else:
            return self._item_cache(idx)