import os
import glob
import json
import io
import random
import itertools
from pathlib import Path
from typing import Tuple, Optional, Callable

import cv2
import numpy as np
from scipy.interpolate import Rbf

from tqdm import tqdm
import webdataset as wds
from google.cloud import storage

from PIL import Image

import torch
from torch.utils.data import Dataset, ConcatDataset, IterableDataset, DataLoader

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
            heatmap_path = os.path.join(
                heatmap_dir, os.path.splitext(file_name)[0] + '.npy')

            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            heatmap = np.load(heatmap_path)

            self.samples.append((image, heatmap))

        # Exclude flagged samples
        self.samples = [s for i, s in enumerate(
            self.samples) if i not in self.exclude_indices]
        print(
            f"Loaded {len(self.samples)} samples after excluding {len(self.exclude_indices)} flagged samples.")

        # Add background samples
        if background_dir:
            # Do glob for PNG files in the background directory
            background_files = glob.glob(os.path.join(background_dir, '*.png'))
            random.shuffle(background_files)

            num_background_samples = min(len(background_files), int(
                len(self.samples) * background_prob))
            print(
                f"Adding {num_background_samples} background samples from {len(background_files)} available files.")

            # Add samples to the dataset
            for i in range(num_background_samples):
                img = cv2.imread(background_files[i])

                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                heatmap = np.zeros(
                    (img.shape[0], img.shape[1]), dtype=np.float32)
                self.samples.append((img, heatmap))

        # Shuffle the dataset
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, heatmap = self.samples[idx]

        # Convert image and heatmap to PIL
        image = Image.fromarray(image)
        heatmap_img = Image.fromarray(
            (heatmap * 255).astype(np.uint8))  # 0–255 grayscale

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
        heatmap_tensor = transforms.ToTensor()(
            heatmap_img).clamp(0, 1).float()  # shape [1, H, W]

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
    train_annotations = [
        ann for ann in annotations if ann["image_id"] in train_img_ids]
    val_annotations = [
        ann for ann in annotations if ann["image_id"] in val_img_ids]

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


class DetectionDataset(Dataset):

    _MAX_NEGATIVE_OFFSET = 100  # Maximum offset for negative samples

    def __init__(
        self,
        tracking_data,
        video_dir="",
        output_size=128,
        crop_size=168,
        interpolation_smoothness=0.5,
        center_offset_range=15,
        temporal_jitter_range=0,
        negative_sample_prob=0.0,
        gaussian_sigma=10.0,
        augmentation_transform=None,
        rotation_range=None,
        scale_range=None,
    ):

        self.video_path = os.path.join(video_dir, tracking_data["video"])
        self.tracking = sorted(
            tracking_data["tracking"], key=lambda x: x["frame_id"])
        self.output_size = output_size
        self.crop_size = crop_size
        self.smooth_trajectory, self.min_frame, self.max_frame = self._interpolate_tracking(
            interpolation_smoothness)
        self.center_offset_range = center_offset_range
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.temporal_jitter_range = temporal_jitter_range
        self.gaussian_sigma = gaussian_sigma
        self.negative_sample_prob = negative_sample_prob

        # Estimate FPS from video file
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            raise Exception(f"Could not open video file: {self.video_path}")

        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = np.array([self.frame_width, self.frame_height])

        self.augmentation_transform = augmentation_transform

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    @staticmethod
    def build_split_datasets(json_filepaths, video_dir, output_size=128, crop_size=256,
                             train_fraction=0.9, center_offset_range=15, temporal_jitter_range=10,
                             negative_sample_prob=0.0, interpolation_smoothness=0.5, gaussian_sigma=6.0,
                             augmentation_transforms=None, rotation_range=None, scale_range=None):

        train_datasets = []
        val_datasets = []

        for path in json_filepaths:
            with open(path, 'r') as f:
                data = json.load(f)

            full_tracking = data['tracking']
            video = data['video']

            # Shuffle and split indices
            indices = list(range(len(full_tracking)))
            random.shuffle(indices)

            split_idx = int(len(indices) * train_fraction)
            train_idx = indices[:split_idx]
            val_idx = indices[split_idx:]

            # Subsets of tracking samples
            train_tracking = [full_tracking[i] for i in train_idx]
            val_tracking = [full_tracking[i] for i in val_idx]

            train_data = {
                "video": video,
                "tracking": train_tracking
            }

            val_data = {
                "video": video,
                "tracking": val_tracking
            }

            # Build train dataset
            train_dataset = DetectionDataset(
                tracking_data=train_data,
                video_dir=video_dir,
                output_size=output_size,
                crop_size=crop_size,
                interpolation_smoothness=interpolation_smoothness,
                temporal_jitter_range=temporal_jitter_range,
                center_offset_range=center_offset_range,
                negative_sample_prob=negative_sample_prob,
                gaussian_sigma=gaussian_sigma,
                rotation_range=rotation_range,
                scale_range=scale_range,
                augmentation_transform=augmentation_transforms
            )
            train_datasets.append(train_dataset)

            # Build val dataset
            val_dataset = DetectionDataset(
                tracking_data=val_data,
                video_dir=video_dir,
                output_size=output_size,
                crop_size=crop_size,
                interpolation_smoothness=interpolation_smoothness,
                temporal_jitter_range=0,
                negative_sample_prob=0.0,
                center_offset_range=0,
                gaussian_sigma=gaussian_sigma
            )
            val_datasets.append(val_dataset)

        # Concat across all videos
        final_train_dataset = ConcatDataset(train_datasets)
        final_val_dataset = ConcatDataset(val_datasets)

        return final_train_dataset, final_val_dataset

    def _interpolate_tracking(self, interpolation_smoothness):
        frame_ids = np.array([pt["frame_id"] for pt in self.tracking])
        coords = np.array([pt["coordinates"] for pt in self.tracking])

        min_frame = int(frame_ids.min())
        max_frame = int(frame_ids.max())
        dense_frames = np.arange(min_frame, max_frame)

        rbf_x = Rbf(
            frame_ids, coords[:, 0], function='multiquadric', epsilon=interpolation_smoothness)
        rbf_y = Rbf(
            frame_ids, coords[:, 1], function='multiquadric', epsilon=interpolation_smoothness)

        x_interp = rbf_x(dense_frames)
        y_interp = rbf_y(dense_frames)

        trajectory = np.stack([x_interp, y_interp], axis=1)

        return trajectory, min_frame, max_frame

    def _read_frame(self, frame_id):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, img = self.video.read()
        if not ret:
            raise Exception(
                f"Could not read frame {frame_id} from video {self.video_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _get_crop(self, frame_id, center, transform: Tuple[float, float, Tuple[float, float]]):
        frame = self._read_frame(frame_id)

        if transform is None:
            x1 = int(center[0] - self.output_size // 2)
            y1 = int(center[1] - self.output_size // 2)
            x2 = x1 + self.output_size
            y2 = y1 + self.output_size

            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                raise IndexError("Crop out of bounds")

            return frame[y1:y2, x1:x2, :]
        else:
            rotation_deg, scale, translate = transform
            crop_size = self.crop_size

            # Compute top-left corner of the large crop
            x1 = int(center[0] - crop_size // 2)
            y1 = int(center[1] - crop_size // 2)
            x2 = x1 + crop_size
            y2 = y1 + crop_size

            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                raise IndexError("Crop out of bounds")

            crop = frame[y1:y2, x1:x2, :]

            center_point = (crop_size // 2, crop_size // 2)
            M = cv2.getRotationMatrix2D(center_point, rotation_deg, scale)

            rotated = cv2.warpAffine(
                crop, M, (crop_size, crop_size), flags=cv2.INTER_LINEAR)

            # Final center crop to self.crop_size
            start = crop_size // 2 - self.output_size // 2
            end = start + self.output_size

            # Apply translation
            x_start = start + int(translate[0])
            x_end = end + int(translate[0])
            y_start = start + int(translate[1])
            y_end = end + int(translate[1])

            return rotated[y_start:y_end, x_start:x_end, :]

    def generate_gaussian_heatmap(self, center):
        x = torch.arange(0, self.output_size, 1).float()
        y = torch.arange(0, self.output_size, 1).float()
        y = y[:, None]

        x0, y0 = center
        heatmap = torch.exp(-((x - x0)**2 + (y - y0)**2) /
                            (2 * self.gaussian_sigma**2))
        return heatmap

    def __len__(self):
        return len(self.tracking)

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def __getitem__(self, idx):
        template_tracking = self.tracking[idx]

        is_negative = random.random() < self.negative_sample_prob

        if not is_negative and (self.rotation_range or self.scale_range):
            rotation_deg = np.random.uniform(-self.rotation_range,
                                             self.rotation_range) if self.rotation_range else 0.0
            scale = np.random.uniform(
                1 - self.scale_range, 1 + self.scale_range) if self.scale_range else 1.0

            if self.center_offset_range > 0:
                offset_x = np.random.uniform(-self.center_offset_range,
                                             self.center_offset_range)
                offset_y = np.random.uniform(-self.center_offset_range,
                                             self.center_offset_range)
                translate = (offset_x, offset_y)
            else:
                translate = (0, 0)

            transform = (rotation_deg, scale, translate)
        else:
            transform = None

        template_frame_id = template_tracking["frame_id"]

        if self.temporal_jitter_range > 0:
            jitter = random.randint(-self.temporal_jitter_range,
                                    self.temporal_jitter_range)
            template_frame_id += jitter

        template_smooth_id = max(
            0, min(template_frame_id - self.min_frame, len(self.smooth_trajectory) - 1))
        template_coordinate = self.smooth_trajectory[template_smooth_id]

        if is_negative:
            offset_x = random.choice([-1, 1]) * random.randint(
                DetectionDataset._MAX_NEGATIVE_OFFSET // 2, DetectionDataset._MAX_NEGATIVE_OFFSET)
            offset_y = random.choice([-1, 1]) * random.randint(
                DetectionDataset._MAX_NEGATIVE_OFFSET // 2, DetectionDataset._MAX_NEGATIVE_OFFSET)

            # Adjust the template coordinates are within the frame
            template_coordinate = (
                max(DetectionDataset._MAX_NEGATIVE_OFFSET // 2, min(
                    template_coordinate[0] + offset_x, self.frame_width - 1 - DetectionDataset._MAX_NEGATIVE_OFFSET // 2)),
                max(DetectionDataset._MAX_NEGATIVE_OFFSET // 2, min(
                    template_coordinate[1] + offset_y, self.frame_height - 1 - DetectionDataset._MAX_NEGATIVE_OFFSET // 2))
            )

        try:
            template = self._get_crop(
                template_frame_id, template_coordinate, transform)
        except IndexError:
            return self.__getitem__((idx + 1) % len(self))

        # Convert to tensor
        template = transforms.ToTensor()(template)

        # Augmentation with same seed
        if not is_negative and self.augmentation_transform:
            template = self.augmentation_transform(
                template.unsqueeze(0)).squeeze(0)

        # Normalize the images
        template = self.normalize(template)

        if is_negative:
            heatmap = torch.zeros((1, self.output_size, self.output_size))
        else:
            center = (self.output_size // 2, self.output_size // 2)

            # Apply random offset to the center
            if self.center_offset_range > 0:
                center = (center[0] - translate[0], center[1] - translate[1])
            heatmap = self.generate_gaussian_heatmap(center).unsqueeze(0)

        return (
            template, heatmap
        )


class CachedDetectionDataset(Dataset):
    def __init__(self, root_dir, output_size=128, gaussian_sigma=15.0):
        with open(os.path.join(root_dir, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)["samples"]

        self.root_dir = root_dir
        self.output_size = output_size
        self.gaussian_sigma = gaussian_sigma

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.metadata)

    def generate_heatmap(self, center):
        x = torch.arange(0, self.output_size).float()
        y = torch.arange(0, self.output_size).float()[:, None]
        x0, y0 = center
        heatmap = torch.exp(-((x - x0)**2 + (y - y0)**2) /
                            (2 * self.gaussian_sigma**2))
        return heatmap.unsqueeze(0)  # Shape: [1, H, W]

    def __getitem__(self, idx):
        item = self.metadata[idx]
        img_path = os.path.join(self.root_dir, item["image_path"])
        img = self.to_tensor(Image.open(img_path).convert("RGB"))
        img = self.normalize(img)

        if item.get("background", False):
            heatmap = torch.zeros((1, self.output_size, self.output_size))
        else:
            heatmap = self.generate_heatmap(item["centroid"])

        return img, heatmap


class TrackingDataset(Dataset):

    _MAX_NEGATIVE_OFFSET = 50  # Maximum offset for negative samples

    def __init__(
        self,
        tracking_data,
        video_dir="",
        output_size=128,
        crop_size=168,
        future_frame_seek=[1, 3, 5, 7],
        random_pick_future_seek=False,
        interpolation_smoothness=0.5,
        temporal_jitter_range=0,
        gaussian_sigma=6.0,
        augmentation_transform=None,
        rotation_range=None,
        scale_range=None,
        negative_sample_prob=0.0,
        centroid_perturbation_range=0.0,
    ):

        self.video_path = os.path.join(video_dir, tracking_data["video"])
        self.tracking = sorted(
            tracking_data["tracking"], key=lambda x: x["frame_id"])
        self.smooth_trajectory, self.min_frame, self.max_frame = self._interpolate_tracking(
            interpolation_smoothness)
        self.future_frame_seek = future_frame_seek
        self.output_size = output_size
        self.crop_size = crop_size
        self.random_pick_future_seek = random_pick_future_seek
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.negative_sample_prob = negative_sample_prob
        self.centroid_perturbation_range = centroid_perturbation_range
        self.temporal_jitter_range = temporal_jitter_range
        self.gaussian_sigma = gaussian_sigma

        # Estimate FPS from video file
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            raise Exception(f"Could not open video file: {self.video_path}")

        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = np.array([self.frame_width, self.frame_height])

        self.augmentation_transform = augmentation_transform

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    @staticmethod
    def build_split_datasets(json_filepaths, video_dir, train_fraction=0.9,
                             future_frame_seek=[1, 3, 5, 7], interpolation_smoothness=0.5, gaussian_sigma=6.0,
                             augmentation_transforms=None, rotation_range=None, scale_range=None, negative_sample_prob=0.0):

        train_datasets = []
        val_datasets = []

        for path in json_filepaths:
            with open(path, 'r') as f:
                data = json.load(f)

            full_tracking = data['tracking']
            video = data['video']

            # Shuffle and split indices
            indices = list(range(len(full_tracking)))
            random.shuffle(indices)

            split_idx = int(len(indices) * train_fraction)
            train_idx = indices[:split_idx]
            val_idx = indices[split_idx:]

            # Subsets of tracking samples
            train_tracking = [full_tracking[i] for i in train_idx]
            val_tracking = [full_tracking[i] for i in val_idx]

            train_data = {
                "video": video,
                "tracking": train_tracking
            }

            val_data = {
                "video": video,
                "tracking": val_tracking
            }

            # Build train dataset
            train_dataset = TrackingDataset(
                tracking_data=train_data,
                video_dir=video_dir,
                future_frame_seek=future_frame_seek,
                random_pick_future_seek=True,
                interpolation_smoothness=interpolation_smoothness,
                gaussian_sigma=gaussian_sigma,
                rotation_range=rotation_range,
                scale_range=scale_range,
                negative_sample_prob=negative_sample_prob,
                augmentation_transform=augmentation_transforms
            )
            train_datasets.append(train_dataset)

            # Build val dataset
            val_dataset = TrackingDataset(
                tracking_data=val_data,
                video_dir=video_dir,
                future_frame_seek=future_frame_seek,
                random_pick_future_seek=False,
                interpolation_smoothness=interpolation_smoothness,
                gaussian_sigma=gaussian_sigma
            )
            val_datasets.append(val_dataset)

        # Concat across all videos
        final_train_dataset = ConcatDataset(train_datasets)
        final_val_dataset = ConcatDataset(val_datasets)

        return final_train_dataset, final_val_dataset

    def _interpolate_tracking(self, interpolation_smoothness):
        frame_ids = np.array([pt["frame_id"] for pt in self.tracking])
        coords = np.array([pt["coordinates"] for pt in self.tracking])

        min_frame = int(frame_ids.min())
        max_frame = int(frame_ids.max())
        dense_frames = np.arange(min_frame, max_frame)

        rbf_x = Rbf(
            frame_ids, coords[:, 0], function='multiquadric', epsilon=interpolation_smoothness)
        rbf_y = Rbf(
            frame_ids, coords[:, 1], function='multiquadric', epsilon=interpolation_smoothness)

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

            cv2.circle(img, (int(coodinate[0]), int(
                coodinate[1])), 5, (0, 255, 0), -1)

            cv2.imshow("Test", np.array(img))
            cv2.waitKey(0)

    def _read_frame(self, frame_id):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, img = self.video.read()
        if not ret:
            raise Exception(
                f"Could not read frame {frame_id} from video {self.video_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _get_crop(self, frame_id, center, transform: Tuple[float, float]):
        frame = self._read_frame(frame_id)

        crop_size = self.crop_size
        output_size = self.output_size

        if transform is None:
            x1 = int(center[0] - output_size // 2)
            y1 = int(center[1] - output_size // 2)
            x2 = x1 + output_size
            y2 = y1 + output_size

            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                raise IndexError("Crop out of bounds")

            return frame[y1:y2, x1:x2, :]
        else:
            rotation_deg, scale = transform
            # Compute top-left corner of the large crop
            x1 = int(center[0] - crop_size // 2)
            y1 = int(center[1] - crop_size // 2)
            x2 = x1 + crop_size
            y2 = y1 + crop_size

            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                raise IndexError("Crop out of bounds")

            crop = frame[y1:y2, x1:x2, :]

            center_point = (crop_size // 2, crop_size // 2)
            M = cv2.getRotationMatrix2D(center_point, rotation_deg, scale)

            rotated = cv2.warpAffine(
                crop, M, (crop_size, crop_size), flags=cv2.INTER_LINEAR)

            # Final center crop to self.crop_size
            start = crop_size // 2 - output_size // 2
            end = start + output_size

            return rotated[start:end, start:end, :]

    def transform_offset_for_heatmap(self, offset, transform: Tuple[float, float]):
        """
        Apply rotation and scale to an offset vector, then map to heatmap coordinates.

        Args:
            offset: np.ndarray shape (2,), the vector (search - template)
            transform: Tuple[float, float] = (rotation_deg, scale)

        Returns:
            np.ndarray of shape (2,), transformed and rescaled offset in heatmap coordinates
        """

        crop_size = self.crop_size
        output_size = self.output_size

        if transform:
            rotation_deg, scale = transform
            theta = np.deg2rad(rotation_deg)

            # 2D rotation matrix with scale
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ]) * scale

            offset = R @ offset

        heatmap_scale = output_size / crop_size
        search_roi_hit = offset * heatmap_scale + output_size // 2

        return search_roi_hit

    def generate_gaussian_heatmap(self, center):
        output_size = self.output_size

        x = torch.arange(0, output_size, 1).float()
        y = torch.arange(0, output_size, 1).float()
        y = y[:, None]

        x0, y0 = center
        heatmap = torch.exp(-((x - x0)**2 + (y - y0)**2) /
                            (2 * self.gaussian_sigma**2))
        return heatmap

    def __len__(self):
        max_future_seek = max(self.future_frame_seek) + \
            self.temporal_jitter_range
        last_frame = self.tracking[-1]["frame_id"]
        num_margin_frames = 0

        for i in range(len(self.tracking) - 1, -1, -1):
            num_margin_frames += 1
            if self.tracking[i]["frame_id"] + max_future_seek < last_frame:
                break

        num_samples = len(self.tracking) - num_margin_frames - 1

        if not self.random_pick_future_seek:
            num_samples *= len(self.future_frame_seek)

        return num_samples

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def __getitem__(self, idx):
        if self.random_pick_future_seek:
            # Reset seed with  time for max randomness
            frame_jump = random.choice(self.future_frame_seek)
            template_tracking = self.tracking[idx]
        else:
            # use modulo to cycle through future_frame_seek
            frame_jump = self.future_frame_seek[idx % len(
                self.future_frame_seek)]
            template_tracking = self.tracking[idx //
                                              len(self.future_frame_seek)]

        if self.rotation_range or self.scale_range:
            rotation_deg = np.random.uniform(-self.rotation_range,
                                             self.rotation_range) if self.rotation_range else 0.0
            scale = np.random.uniform(
                1 - self.scale_range, 1 + self.scale_range) if self.scale_range else 1.0
            transform = (rotation_deg, scale)
        else:
            transform = None

        template_frame_id = template_tracking["frame_id"]

        if self.temporal_jitter_range > 0:
            jitter = random.randint(-self.temporal_jitter_range,
                                    self.temporal_jitter_range)
            template_frame_id += jitter

        search_frame_id = template_frame_id + frame_jump

        template_smooth_id = template_frame_id - self.min_frame
        search_smooth_id = template_smooth_id + frame_jump

        template_coordinate = self.smooth_trajectory[template_smooth_id]
        search_coordinate = self.smooth_trajectory[search_smooth_id]

        if self.centroid_perturbation_range > 0.0:
            perturbation_x = np.random.uniform(
                -self.centroid_perturbation_range, self.centroid_perturbation_range)
            perturbation_y = np.random.uniform(
                -self.centroid_perturbation_range, self.centroid_perturbation_range)
            template_coordinate = (
                template_coordinate[0] + perturbation_x, template_coordinate[1] + perturbation_y)

        is_negative = random.random() < self.negative_sample_prob

        if is_negative:
            offset_x = random.choice([-1, 1]) * random.randint(
                TrackingDataset._MAX_NEGATIVE_OFFSET // 2, TrackingDataset._MAX_NEGATIVE_OFFSET)
            offset_y = random.choice([-1, 1]) * random.randint(
                TrackingDataset._MAX_NEGATIVE_OFFSET // 2, TrackingDataset._MAX_NEGATIVE_OFFSET)

            template_coordinate = (
                search_coordinate[0] + offset_x,
                search_coordinate[1] + offset_y
            )

            max_frame_seek = max(self.future_frame_seek)
            search_frame_id = search_smooth_id + \
                random.randint(-max_frame_seek, max_frame_seek)

        try:
            template = self._get_crop(
                template_frame_id, template_coordinate, transform)
            search = self._get_crop(
                search_frame_id, template_coordinate, transform)
        except IndexError:
            return self.__getitem__((idx + 1) % len(self))

        to_tensor = transforms.ToTensor()
        template = to_tensor(template)
        search = to_tensor(search)

        # Transform/augment both images with same function.
        if self.augmentation_transform:
            stacked = torch.stack([template, search])
            transformed = self.augmentation_transform(stacked)
            template, search = transformed[0], transformed[1]

        # Normalize the images
        template = self.normalize(template)
        search = self.normalize(search)

        if is_negative:
            heatmap = torch.zeros(
                (1, self.output_size, self.output_size))
        else:
            offset = np.array(search_coordinate) - \
                np.array(template_coordinate)
            search_roi_hit = self.transform_offset_for_heatmap(
                offset, transform)
            heatmap = self.generate_gaussian_heatmap(
                search_roi_hit).unsqueeze(0)

        return (
            template, search, heatmap
        )


class CachedTrackingDataset(Dataset):
    """
    Cached version of TrackingDataset.
    Loads crops and metadata from disk, avoiding video decoding at runtime.
    Each sample contains (template, search, heatmap).
    """

    def __init__(self, root_dir, output_size=128, gaussian_sigma=6.0):
        with open(os.path.join(root_dir, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)["samples"]

        self.root_dir = root_dir
        self.output_size = output_size
        self.gaussian_sigma = gaussian_sigma

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.metadata)

    def generate_heatmap(self, center):
        x = torch.arange(0, self.output_size).float()
        y = torch.arange(0, self.output_size).float()[:, None]
        x0, y0 = center
        heatmap = torch.exp(-((x - x0)**2 + (y - y0)**2) /
                            (2 * self.gaussian_sigma**2))
        return heatmap.unsqueeze(0)  # Shape: [1, H, W]

    def __getitem__(self, idx):
        item = self.metadata[idx]
        template_path = os.path.join(
            self.root_dir, item["template_path"])
        search_path = os.path.join(self.root_dir, item["search_path"])
        template = self.to_tensor(
            Image.open(template_path).convert("RGB"))
        search = self.to_tensor(Image.open(search_path).convert("RGB"))
        template = self.normalize(template)
        search = self.normalize(search)

        if item.get("background", False):
            heatmap = torch.zeros(
                (1, self.output_size, self.output_size))
        else:
            heatmap = self.generate_heatmap(item["centroid"])

        return template, search, heatmap


class BackgroundWindowDataset(torch.utils.data.Dataset):
    """
    Dataset for sampling background (no-fish) windows from a video, using a sliding window approach.
    The user provides a list of frame indices known to contain only background (no fish present).
    Each sample is a cropped window from a background frame, with optional augmentation, rotation, and scaling.
    The output is (image, heatmap), where heatmap is always a zero tensor.
    """

    def __init__(
        self,
        video_path: str,
        background_frames: list[int],
        output_size: int = 128,
        crop_size: int = 168,
        siamese: bool = False,
        stride_portion: float = 0.5,
        augmentation_transform=None,
        rotation_range=None,
        scale_range=None,
    ):
        self.video_path = video_path
        self.background_frames = sorted(background_frames)
        self.output_size = output_size
        self.crop_size = crop_size
        self.siamese = siamese
        self.stride_portion = stride_portion
        self.augmentation_transform = augmentation_transform
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        # Open video and get frame size
        self.video = cv2.VideoCapture(self.video_path)
        if not self.video.isOpened():
            raise Exception(f"Could not open video file: {self.video_path}")
        self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Precompute all valid (frame, y, x) window positions
        self.samples = []
        stride = max(1, int(self.stride_portion * self.output_size))
        for frame_id in self.background_frames:
            for y in range(0, self.frame_height - self.crop_size + 1, stride):
                for x in range(0, self.frame_width - self.crop_size + 1, stride):
                    self.samples.append((frame_id, y, x))

    def __len__(self):
        return len(self.samples)

    def __del__(self):
        if hasattr(self, 'video') and self.video.isOpened():
            self.video.release()

    def _read_frame(self, frame_id):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, img = self.video.read()
        if not ret:
            raise Exception(
                f"Could not read frame {frame_id} from video {self.video_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        frame_id, y, x = self.samples[idx]
        # Optionally apply rotation/scale
        if self.rotation_range or self.scale_range:
            rotation_deg = np.random.uniform(-self.rotation_range,
                                             self.rotation_range) if self.rotation_range else 0.0
            scale = np.random.uniform(
                1 - self.scale_range, 1 + self.scale_range) if self.scale_range else 1.0
        else:
            rotation_deg = 0.0
            scale = 1.0
        # Read frame and crop
        frame = self._read_frame(frame_id)
        crop = frame[y:y+self.crop_size, x:x+self.crop_size, :]
        # Apply rotation/scale if needed
        if rotation_deg != 0.0 or scale != 1.0:
            center_point = (self.crop_size // 2, self.crop_size // 2)
            M = cv2.getRotationMatrix2D(center_point, rotation_deg, scale)
            crop = cv2.warpAffine(
                crop, M, (self.crop_size, self.crop_size), flags=cv2.INTER_LINEAR)
        # Final center crop to output_size
        start = self.crop_size // 2 - self.output_size // 2
        end = start + self.output_size
        crop = crop[start:end, start:end, :]
        # To tensor
        crop = transforms.ToTensor()(crop)
        # Augmentation
        if self.augmentation_transform:
            crop = self.augmentation_transform(crop.unsqueeze(0)).squeeze(0)
        crop = self.normalize(crop)
        heatmap = torch.zeros((1, self.output_size, self.output_size))

        if self.siamese:
            # For Siamese networks, return two identical crops
            return crop, crop, heatmap
        else:
            return crop, heatmap


class WebTrackingDataset(IterableDataset):
    """
    Webdataset-based tracking dataset.
    
    Reads template/search image pairs and metadata from tar files.
    Each sample contains (template, search, heatmap).
    
    Args:
        tar_urls: Path or list of paths to tar files (can include wildcards)
                  Examples: 
                    - "path/to/tracking-{000000..000009}.tar"
                    - ["path/to/shard1.tar", "path/to/shard2.tar"]
                    - "s3://bucket/tracking-*.tar"
        output_size: Size of output heatmap (default: 128)
        gaussian_sigma: Sigma for Gaussian heatmap generation (default: 6.0)
        shuffle: Whether to shuffle samples (default: True)
        transform: Optional transform to apply to images
        empty_check: If True, raises an error if no samples are found in shards
    """
    
    def __init__(
        self,
        tar_urls,
        output_size: int = 128,
        gaussian_sigma: float = 6.0,
        shuffle: bool = True,
        transform: Optional[Callable] = None,
        empty_check: bool = False,
        max_samples: Optional[int] = None
    ):
        self.tar_urls = tar_urls
        self.output_size = output_size
        self.gaussian_sigma = gaussian_sigma
        self.shuffle = shuffle
        self.custom_transform = transform
        self.empty_check = empty_check
        self.max_samples = max_samples
        
        # Standard image preprocessing
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def generate_heatmap(self, center):
        """Generate Gaussian heatmap centered at the given position."""
        x = torch.arange(0, self.output_size).float()
        y = torch.arange(0, self.output_size).float()[:, None]
        x0, y0 = center
        heatmap = torch.exp(-((x - x0)**2 + (y - y0)**2) /
                            (2 * self.gaussian_sigma**2))
        return heatmap.unsqueeze(0)  # Shape: [1, H, W]
    
    def decode_sample(self, sample):
        """
        Decode a webdataset sample into (template, search, heatmap) tuple.
        
        Args:
            sample: Dictionary containing keys like 'template.jpg', 'search.jpg', 'json'
        
        Returns:
            Tuple of (template_tensor, search_tensor, heatmap_tensor)
        """
        # Load and preprocess template image
        template_bytes = sample['template.jpg']
        template = Image.open(io.BytesIO(template_bytes)).convert("RGB")
        template = self.to_tensor(template)
        template = self.normalize(template)
        
        # Load and preprocess search image
        search_bytes = sample['search.jpg']
        search = Image.open(io.BytesIO(search_bytes)).convert("RGB")
        search = self.to_tensor(search)
        search = self.normalize(search)
        
        # Load metadata
        metadata = json.loads(sample['json'])
        
        # Generate heatmap
        if metadata.get("background", False):
            heatmap = torch.zeros((1, self.output_size, self.output_size))
        else:
            heatmap = self.generate_heatmap(metadata["centroid"])
        
        return template, search, heatmap
    
    def __iter__(self):
        """Create an iterator over the dataset."""
        # Create webdataset pipeline
        dataset = wds.WebDataset(self.tar_urls, shardshuffle=False, empty_check=self.empty_check)
        
        # Optionally shuffle
        if self.shuffle:
            dataset = dataset.shuffle(1000)  # Shuffle buffer of 1000 samples
        
        # Map to our format
        dataset = dataset.map(self.decode_sample)
        
        # Limit samples AFTER decoding to ensure hard limit per worker
        if self.max_samples is not None:
            # Get worker info to divide samples among workers
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # Multiple workers: each worker gets a portion of max_samples
                per_worker = int(np.ceil(self.max_samples / worker_info.num_workers))
                return itertools.islice(iter(dataset), per_worker)
            else:
                # Single worker: use all max_samples
                return itertools.islice(iter(dataset), self.max_samples)
        
        return iter(dataset)


class WebTrackingDatasetWithLength(WebTrackingDataset):
    """
    Extended version with approximate length for DataLoader compatibility.
    
    This is useful when you need a DataLoader with a known length for
    progress bars or epoch-based training.
    
    Args:
        tar_urls: Path or list of paths to tar files
        length: Total number of samples in the dataset
        output_size: Size of output heatmap (default: 128)
        gaussian_sigma: Sigma for Gaussian heatmap generation (default: 6.0)
        shuffle: Whether to shuffle samples (default: True)
        transform: Optional transform to apply to images
    """
    
    def __init__(
        self,
        tar_urls,
        length: int,
        output_size: int = 128,
        gaussian_sigma: float = 6.0,
        shuffle: bool = True,
        transform: Optional[Callable] = None,
        empty_check: bool = False
    ):
        super().__init__(tar_urls, output_size, gaussian_sigma, shuffle, transform, empty_check, max_samples=length)
        self._length = length
    
    def __len__(self):
        """Return the approximate length of the dataset."""
        return self._length

def fetch_shard_dataset(output_dir: str, verbose=True):
    """Fetch the shards dataset from GCS to a given directory"""

    # If output_dir does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if we have write rights on this directory and that its empty
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Cannot write to directory: {output_dir}")
    
    if os.listdir(output_dir):
        raise FileExistsError(f"Directory is not empty: {output_dir}")
    
    gcs_shards_dir = "shards"

    # Initialize a guest client

    client = storage.Client.create_anonymous_client()

    bucket = client.bucket("aegear-training-data")
    blobs = bucket.list_blobs(prefix=gcs_shards_dir)
    blobs = tqdm(blobs, desc="Downloading shards") if verbose else blobs

    # Tarballs and the manifest file
    for blob in blobs: 
        if not blob.name.endswith(".tar") and not blob.name.endswith(".json"):
            continue

        # Download to output_dir
        destination_path = os.path.join(output_dir, os.path.basename(blob.name))
        blob.download_to_filename(destination_path)

def create_webdataset_from_manifest(
    manifest_path: str,
    output_size: int = 128,
    gaussian_sigma: float = 6.0,
    shuffle: bool = True,
    autodownload: bool = True,
    verbose: bool = True
) -> WebTrackingDatasetWithLength:
    """
    Create a WebTrackingDataset from a manifest file.
    
    Args:
        manifest_path: Path to the manifest JSON file
        output_size: Size of output heatmap
        gaussian_sigma: Sigma for Gaussian heatmap generation
        shuffle: Whether to shuffle samples
        autodownload: Whether to auto-download shards if not present
        verbose: Whether to print download progress
    
    Raises:
        ValueError: If number of tar files does not match num_shards in manifest.
    
    Returns:
        WebTrackingDatasetWithLength instance.
    """
    # Get directory of manifest to build tar file paths
    data_dir = os.path.dirname(manifest_path)

    # If manifest or tar files do not exist, auto-download
    if autodownload and (not os.path.exists(manifest_path) or not any(Path(data_dir).glob("*.tar"))):
        if verbose:
            print("Manifest or tar files not found. Auto-downloading shards...")
        fetch_shard_dataset(data_dir, verbose=verbose)

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    
    # Check if the same directory contains the tar files
    tar_files = sorted(Path(data_dir).glob("*.tar"))
    tar_files = [str(f) for f in tar_files]

    num_shards = manifest['num_shards']
    total_samples = manifest['total_samples']

    # Check num_shards matching with tar files found
    if len(tar_files) != num_shards:
        raise ValueError(
            f"Number of tar files found ({len(tar_files)}) does not match num_shards in manifest ({num_shards}).")
    
    # Build the URL pattern for webdataset
    # This assumes all shards follow the pattern in the manifest
    prefix = manifest['shard_pattern'].split('-')[0]
    tar_pattern = os.path.join(data_dir, f"{prefix}-{{000000..{num_shards-1:06d}}}.tar")
    
    return WebTrackingDatasetWithLength(
        tar_urls=tar_pattern,
        length=total_samples,
        output_size=output_size,
        gaussian_sigma=gaussian_sigma,
        shuffle=shuffle
    )

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_shards_train_val(
    shard_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple:
    """
    Split tar files into train and validation sets with a predictable seed.
    
    Args:
        shard_dir: Directory containing the tar files
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_tar_files, val_tar_files)
    """
    # Get all tar files
    tar_files = sorted(Path(shard_dir).glob("*.tar"))
    tar_files = [str(f) for f in tar_files]
    
    # Set seed for reproducible splitting
    random.seed(seed)
    random.shuffle(tar_files)
    
    # Split
    split_idx = int(len(tar_files) * train_ratio)
    train_files = tar_files[:split_idx]
    val_files = tar_files[split_idx:]
    
    print(f"Split {len(tar_files)} shards into:")
    print(f"  Training: {len(train_files)} shards")
    print(f"  Validation: {len(val_files)} shards")
    
    return train_files, val_files


def calculate_approximate_samples(tar_files: list, manifest_path: str = None) -> int:
    """
    Calculate approximate number of samples from tar files.
    
    If manifest is available, use it for accurate count.
    Otherwise, estimate based on number of shards.
    """
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        total_samples = manifest['total_samples']
        num_shards = manifest['num_shards']
        
        # Calculate samples for the given tar files
        samples_per_shard = total_samples / num_shards
        return int(len(tar_files) * samples_per_shard)
    else:
        # Rough estimate: assume 5000 samples per shard (default shard size)
        return len(tar_files) * 5000

def create_webdataset_from_manifest(
    manifest_path: str,
    output_size: int = 128,
    gaussian_sigma: float = 6.0,
    train_ratio: float = 0.8,
    seed: int = 42,
    autodownload: bool = True,
    verbose: bool = True
) -> Tuple[WebTrackingDatasetWithLength, WebTrackingDatasetWithLength]:
    """
    Create train and validation WebTrackingDatasets from a manifest file.
    
    Args:
        manifest_path: Path to the manifest JSON file
        output_size: Size of output heatmap
        gaussian_sigma: Sigma for Gaussian heatmap generation
        train_ratio: Ratio of data to use for training (default: 0.8)
        seed: Random seed for reproducible splitting (default: 42)
        autodownload: Whether to auto-download shards if not present
        verbose: Whether to print download progress
    
    Returns:
        Tuple of (train_dataset, val_dataset) as WebTrackingDatasetWithLength instances
    
    Raises:
        ValueError: If number of tar files does not match num_shards in manifest
    """
    
    # Get directory of manifest to build tar file paths
    data_dir = os.path.dirname(manifest_path)

    # If manifest or tar files do not exist, auto-download
    if autodownload and (not os.path.exists(manifest_path) or not any(Path(data_dir).glob("*.tar"))):
        if verbose:
            print("Manifest or tar files not found. Auto-downloading shards...")
        fetch_shard_dataset(data_dir, verbose=verbose)

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Check if the same directory contains the tar files
    tar_files = sorted(Path(data_dir).glob("*.tar"))
    tar_files = [str(f) for f in tar_files]
    
    num_shards = manifest['num_shards']
    total_samples = manifest['total_samples']
    
    # Check num_shards matching with tar files found
    if len(tar_files) != num_shards:
        raise ValueError(
            f"Number of tar files found ({len(tar_files)}) does not match num_shards in manifest ({num_shards})")
    
    # Split tar files into train/val with seed
    random.seed(seed)
    random.shuffle(tar_files)
    
    split_idx = int(len(tar_files) * train_ratio)
    train_files = tar_files[:split_idx]
    val_files = tar_files[split_idx:]
    
    print(f"Split {len(tar_files)} shards into:")
    print(f"  Training: {len(train_files)} shards")
    print(f"  Validation: {len(val_files)} shards")
    
    # Calculate approximate samples per split
    samples_per_shard = total_samples / num_shards
    train_samples = int(len(train_files) * samples_per_shard)
    val_samples = int(len(val_files) * samples_per_shard)
    
    print(f"  Approximate samples - Train: {train_samples}, Val: {val_samples}")
    
    # Create train dataset
    train_dataset = WebTrackingDatasetWithLength(
        tar_urls=train_files,
        length=train_samples,
        output_size=output_size,
        gaussian_sigma=gaussian_sigma,
        shuffle=True,  # Shuffle training data
        empty_check=False
    )
    
    # Create validation dataset
    val_dataset = WebTrackingDatasetWithLength(
        tar_urls=val_files,
        length=val_samples,
        output_size=output_size,
        gaussian_sigma=gaussian_sigma,
        shuffle=False,  # Don't shuffle validation data
        empty_check=False
    )
    
    return train_dataset, val_dataset

def load_dataset_from_shards(manifest_path: str,
                             output_size: int = 128,
                             gaussian_sigma: float = 6.0,
                             batch_size: int = 128,
                             train_ratio: float = 0.8,
                             num_workers: int = 0,
                             seed: int = 42,
                             autodownload: bool = True,
                             verbose: bool = True) -> tuple:
    """
    Load training and validation datasets from tar shards.
    
    Args:
        manifest_path: Path to the manifest JSON file
        output_size: Size of output heatmap
        gaussian_sigma: Sigma for Gaussian heatmap generation
        batch_size: Batch size for DataLoader
        train_ratio: Ratio of data to use for training
        num_workers: Number of DataLoader workers
        seed: Random seed for reproducibility
        autodownload: Whether to auto-download shards if not present
        verbose: Whether to print download progress
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_dataset, val_dataset = create_webdataset_from_manifest(
        manifest_path=manifest_path,
        output_size=output_size,
        gaussian_sigma=gaussian_sigma,
        train_ratio=train_ratio,
        seed=seed,
        autodownload=autodownload,
        verbose=verbose
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_loader, val_loader

if __name__ == "__main__":
    """Temporary test - to be deleted later."""

    # Set seed for reproducibility
    set_seed(42)

    data_dir = "data/training/tracking_merged"
    batch_size = 128
    
    # Split shards into train/val
    train_files, val_files = split_shards_train_val(
        data_dir,
        train_ratio=0.8,
        seed=42
    )
    
    # Create datasets
    train_dataset = WebTrackingDataset(
        tar_urls=train_files,
        output_size=128,
        gaussian_sigma=6.0,
        shuffle=True
    )
    
    val_dataset = WebTrackingDataset(
        tar_urls=val_files,
        output_size=128,
        gaussian_sigma=6.0,
        shuffle=False  # Don't shuffle validation
    )
    
    # Create dataloaders
    # Note: num_workers > 0 with IterableDataset requires careful setup
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0  # Start with 0, can increase if needed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0
    )