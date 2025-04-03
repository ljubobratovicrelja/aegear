import os
import glob
import json
import random

import cv2
import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset


class FishHeatmapDataset(Dataset):
    def __init__(self, annotation_json, image_dir, heatmap_dir,
                 background_dir=None, background_prob=0.3,
                 img_transform=None, joint_transform=None,
                 exclude_indices=None):
        self.samples = []
        self.img_transform = img_transform
        self.joint_transform = joint_transform
        self.exclude_indices = set(exclude_indices or [])

        with open(annotation_json, 'r') as f:
            coco_data = json.load(f)

        for img_info in coco_data['images']:
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

        # Add background samples
        if background_dir:
            # Do glob for PNG files in the background directory
            background_files = glob.glob(os.path.join(background_dir, '*.png'))
            random.shuffle(background_files)

            num_background_samples = min(len(background_files), int(len(self.samples) * background_prob))

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
        image, heatmap  = self.samples[idx]

        # Convert image to PIL (color image)
        image = Image.fromarray(image)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)  # [1, H, W]

        # Apply same random transforms to both
        if self.joint_transform:
            seed = np.random.randint(0, 10000)
            torch.manual_seed(seed)
            image = self.joint_transform(image)
            torch.manual_seed(seed)
            heatmap_tensor = self.joint_transform(heatmap_tensor)

        # Individual transforms (image normalization etc.)
        if self.img_transform:
            image = self.img_transform(image)

        # Clamp heatmap just in case and ensure float32
        heatmap_tensor = heatmap_tensor.clamp(0, 1).float()

        return image, heatmap_tensor