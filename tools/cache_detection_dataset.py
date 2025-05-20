import os
import json
import random
from urllib.request import urlretrieve
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import ConcatDataset

# adjust import as needed
from aegear.dataset import DetectionDataset, BackgroundWindowDataset
from aegear.transforms import RandomPoissonNoise  # adjust if used

# Paths
dataset_dir = "../data/training"
video_dir = "../data/video"
cache_dir = "../data/cache"

public_base_url = "https://storage.googleapis.com/aegear-training-data"

annotations = {
    "E7": {...},  # keep your existing entries here
    "K9": {...},
    "S1": {...},
    "4_per_23": {...},
    "5_per_12": {...}
}

# Ensure dirs
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# Download annotations and videos
for key, ann in tqdm(annotations.items(), desc="Downloading"):
    ann_path = os.path.join(dataset_dir, ann["file"])
    vid_path = os.path.join(video_dir, f"{key}.MOV")

    if not os.path.exists(ann_path):
        print(f"Downloading {ann['annotation_url']}")
        urlretrieve(ann["annotation_url"], ann_path)

    if not os.path.exists(vid_path):
        print(f"Downloading {ann['video_url']}")
        urlretrieve(ann["video_url"], vid_path)

# Prepare dataset
augmentation_transforms = transforms.Compose([
    RandomPoissonNoise(p=0.15),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.05, 0.75)),
    transforms.ColorJitter(brightness=0.35, contrast=0.25,
                           saturation=0.25, hue=0.15),
])

interpolation_smoothness = 5.0
train_fraction = 0.9
output_size = 128
crop_size = 256
cache_multiplier = 4

annotation_paths = [os.path.join(dataset_dir, a["file"])
                    for a in annotations.values()]
train_dataset, val_dataset = DetectionDataset.build_split_datasets(
    annotation_paths, video_dir,
    output_size=output_size,
    crop_size=crop_size,
    train_fraction=train_fraction,
    temporal_jitter_range=5,
    center_offset_range=30,
    negative_sample_prob=0.0,
    gaussian_sigma=15.0,
    interpolation_smoothness=interpolation_smoothness,
    augmentation_transforms=augmentation_transforms,
    rotation_range=30.0,
    scale_range=0.3
)

# Cache dataset


def cache_dataset(dataset, name):
    meta = []
    os.makedirs(os.path.join(cache_dir, name, "crops"), exist_ok=True)
    sample_id = 0
    for repeat in range(cache_multiplier):
        for idx in tqdm(range(len(dataset)), desc=f"Caching {name} set, pass {repeat+1}"):
            try:
                img, heatmap = dataset[idx]
                img_np = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
                # you can adjust this if offset applied
                center = (output_size // 2, output_size // 2)

                filename = f"{sample_id:06d}.jpg"
                filepath = os.path.join(cache_dir, name, "crops", filename)
                from PIL import Image
                Image.fromarray(img_np).save(filepath)

                meta.append({
                    "frame_id": idx,
                    "crop_path": f"crops/{filename}",
                    "center_xy": center,
                    "is_negative": False
                })

                sample_id += 1
            except Exception as e:
                print(f"Failed at idx {idx}: {e}")
                continue

    with open(os.path.join(cache_dir, f"{name}_metadata.json"), "w") as f:
        json.dump({"samples": meta}, f, indent=2)


cache_dataset(train_dataset, "train")
cache_dataset(val_dataset, "val")
