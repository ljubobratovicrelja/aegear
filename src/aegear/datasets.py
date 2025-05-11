import os
import glob
import json
import random
from pathlib import Path
from typing import Tuple
from collections import OrderedDict

import cv2
import numpy as np
from scipy.interpolate import Rbf

from PIL import Image

import torch
from torch.utils.data import Dataset, ConcatDataset

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

class DetectionDataset(Dataset):

    _MAX_NEGATIVE_OFFSET = 100 # Maximum offset for negative samples

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
        self.tracking = sorted(tracking_data["tracking"], key=lambda x: x["frame_id"])
        self.output_size = output_size
        self.crop_size = crop_size
        self.smooth_trajectory, self.min_frame, self.max_frame = self._interpolate_tracking(interpolation_smoothness)
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

        rbf_x = Rbf(frame_ids, coords[:, 0], function='multiquadric', epsilon=interpolation_smoothness)
        rbf_y = Rbf(frame_ids, coords[:, 1], function='multiquadric', epsilon=interpolation_smoothness)

        x_interp = rbf_x(dense_frames)
        y_interp = rbf_y(dense_frames)

        trajectory = np.stack([x_interp, y_interp], axis=1)

        return trajectory, min_frame, max_frame
    
    def _read_frame(self, frame_id):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, img = self.video.read()
        if not ret:
            raise Exception(f"Could not read frame {frame_id} from video {self.video_path}")
        
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

            rotated = cv2.warpAffine(crop, M, (crop_size, crop_size), flags=cv2.INTER_LINEAR)

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
        heatmap = torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.gaussian_sigma**2))
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
            rotation_deg = np.random.uniform(-self.rotation_range, self.rotation_range) if self.rotation_range else 0.0
            scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range) if self.scale_range else 1.0

            if self.center_offset_range > 0:
                offset_x = np.random.uniform(-self.center_offset_range, self.center_offset_range)
                offset_y = np.random.uniform(-self.center_offset_range, self.center_offset_range)
                translate = (offset_x, offset_y)
            else:
                translate = (0, 0)

            transform = (rotation_deg, scale, translate)
        else:
            transform = None

        template_frame_id = template_tracking["frame_id"]

        if self.temporal_jitter_range > 0:
            jitter = random.randint(-self.temporal_jitter_range, self.temporal_jitter_range)
            template_frame_id += jitter

        template_smooth_id = max(0, min(template_frame_id - self.min_frame, len(self.smooth_trajectory) - 1))
        template_coordinate = self.smooth_trajectory[template_smooth_id]

        if is_negative:
            offset_x = random.choice([-1, 1]) * random.randint(DetectionDataset._MAX_NEGATIVE_OFFSET // 2, DetectionDataset._MAX_NEGATIVE_OFFSET)
            offset_y = random.choice([-1, 1]) * random.randint(DetectionDataset._MAX_NEGATIVE_OFFSET // 2, DetectionDataset._MAX_NEGATIVE_OFFSET)

            # Adjust the template coordinates are within the frame
            template_coordinate = (
                max(DetectionDataset._MAX_NEGATIVE_OFFSET // 2, min(template_coordinate[0] + offset_x, self.frame_width - 1 - DetectionDataset._MAX_NEGATIVE_OFFSET // 2)),
                max(DetectionDataset._MAX_NEGATIVE_OFFSET // 2, min(template_coordinate[1] + offset_y, self.frame_height - 1 - DetectionDataset._MAX_NEGATIVE_OFFSET // 2))
            )

        try:
            template = self._get_crop(template_frame_id, template_coordinate, transform)
        except IndexError:
            return self.__getitem__((idx + 1) % len(self))

        # Convert to tensor
        template = transforms.ToTensor()(template)

        # Augmentation with same seed
        if not is_negative and self.augmentation_transform:
            template = self.augmentation_transform(template.unsqueeze(0)).squeeze(0)

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
    
        
class TrackingDataset(Dataset):

    _MAX_NEGATIVE_OFFSET = 50 # Maximum offset for negative samples
    _OUTPUT_SIZE = 128
    _CROP_SIZE = 168

    def __init__(
        self,
        tracking_data,
        video_dir="",
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
        self.tracking = sorted(tracking_data["tracking"], key=lambda x: x["frame_id"])
        self.smooth_trajectory, self.min_frame, self.max_frame = self._interpolate_tracking(interpolation_smoothness)
        self.future_frame_seek = future_frame_seek
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

    def _get_crop(self, frame_id, center, transform: Tuple[float, float]):
        frame = self._read_frame(frame_id)

        crop_size = TrackingDataset._CROP_SIZE
        output_size = TrackingDataset._OUTPUT_SIZE

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

            rotated = cv2.warpAffine(crop, M, (crop_size, crop_size), flags=cv2.INTER_LINEAR)

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

        crop_size = TrackingDataset._CROP_SIZE
        output_size = TrackingDataset._OUTPUT_SIZE

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

    def generate_gaussian_heatmap(self, center ):
        output_size = TrackingDataset._OUTPUT_SIZE

        x = torch.arange(0, output_size, 1).float()
        y = torch.arange(0, output_size, 1).float()
        y = y[:, None]

        x0, y0 = center
        heatmap = torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.gaussian_sigma**2))
        return heatmap

    def __len__(self):
        max_future_seek = max(self.future_frame_seek) + self.temporal_jitter_range
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
            frame_jump = self.future_frame_seek[idx % len(self.future_frame_seek)]
            template_tracking = self.tracking[idx // len(self.future_frame_seek)]

        if self.rotation_range or self.scale_range:
            rotation_deg = np.random.uniform(-self.rotation_range, self.rotation_range) if self.rotation_range else 0.0
            scale = np.random.uniform(1 - self.scale_range, 1 + self.scale_range) if self.scale_range else 1.0
            transform = (rotation_deg, scale)
        else:
            transform = None

        template_frame_id = template_tracking["frame_id"]

        if self.temporal_jitter_range > 0:
            jitter = random.randint(-self.temporal_jitter_range, self.temporal_jitter_range)
            template_frame_id += jitter

        search_frame_id = template_frame_id + frame_jump

        template_smooth_id = template_frame_id - self.min_frame
        search_smooth_id = template_smooth_id + frame_jump

        template_coordinate = self.smooth_trajectory[template_smooth_id]
        search_coordinate = self.smooth_trajectory[search_smooth_id]

        if self.centroid_perturbation_range > 0.0:
            perturbation_x = np.random.uniform(-self.centroid_perturbation_range, self.centroid_perturbation_range)
            perturbation_y = np.random.uniform(-self.centroid_perturbation_range, self.centroid_perturbation_range)
            template_coordinate = (template_coordinate[0] + perturbation_x, template_coordinate[1] + perturbation_y)
                   
        is_negative = random.random() < self.negative_sample_prob

        if is_negative:
            offset_x = random.choice([-1, 1]) * random.randint(TrackingDataset._MAX_NEGATIVE_OFFSET // 2, TrackingDataset._MAX_NEGATIVE_OFFSET)
            offset_y = random.choice([-1, 1]) * random.randint(TrackingDataset._MAX_NEGATIVE_OFFSET // 2, TrackingDataset._MAX_NEGATIVE_OFFSET)

            template_coordinate = (
                search_coordinate[0] + offset_x,
                search_coordinate[1] + offset_y
            )

            max_frame_seek = max(self.future_frame_seek)
            search_frame_id = search_smooth_id + random.randint(-max_frame_seek, max_frame_seek)

        try:
            template = self._get_crop(template_frame_id, template_coordinate, transform)
            search = self._get_crop(search_frame_id, template_coordinate, transform)
        except IndexError:
            return self.__getitem__((idx + 1) % len(self))
        
        to_tensor = transforms.ToTensor()
        template = to_tensor(template)
        search = to_tensor(search)

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

        if is_negative:
            heatmap = torch.zeros((1, TrackingDataset._OUTPUT_SIZE, TrackingDataset._OUTPUT_SIZE))
        else:
            offset = np.array(search_coordinate) - np.array(template_coordinate)
            search_roi_hit = self.transform_offset_for_heatmap(offset, transform)
            heatmap = self.generate_gaussian_heatmap(search_roi_hit).unsqueeze(0)

        return (
            template, search, heatmap
        )
    
        