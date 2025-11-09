import os

import zipfile
from google.cloud import storage

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from tqdm import tqdm

import numpy as np

import fiftyone as fo
import fiftyone.utils.torch as fou

from aegear.model import SiameseTracker
from aegear.datasets import CachedTrackingDataset
from aegear.training import get_centroids_per_sample
from aegear.utils import get_latest_model_path


def download_dataset(dataset_dir):
    bucket_name = "aegear-training-data"
    blob_path = "cache/tracking.zip"

    if os.path.exists(os.path.join(dataset_dir, "tracking")):
        print("Dataset already exists. Skipping download.")
    else:
        print("Dataset not found. Downloading...")
        zip_path = os.path.join(dataset_dir, "tracking.zip")
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

# --- 1. Configuration ---
DATASET_DIR = "data/training/cache"
DATASET_NAME = "4_per_23"
MODELS_DIR = "models/"
GAUSSIAN_SIGMA = 12.0
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    # Ensure dataset is available
    download_dataset(DATASET_DIR)

    # Load Model and Data
    print("Loading model and dataset...")

    model_path = get_latest_model_path(MODELS_DIR, "model_siamese")

    model = SiameseTracker()  # Assumes default EfficientUNet
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load Dataset and DataLoader
    # IMPORTANT: shuffle=False is critical to map predictions back
    val_dataset = CachedTrackingDataset(os.path.join(DATASET_DIR, "tracking", DATASET_NAME, "val"), gaussian_sigma=GAUSSIAN_SIGMA)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Must be False
        num_workers=4
    )
    
    img_size = val_dataset.output_size

    # Populate FiftyOne Dataset from Metadata
    print("Populating FiftyOne dataset from metadata...")
    dataset = fo.Dataset("siamese-tracker-eval", persistent=True, overwrite=True)

    samples_to_add = []
    for item in tqdm(val_dataset.metadata, desc="Loading metadata"):
        # The "filepath" is the search image
        search_path = os.path.join(val_dataset.root_dir, item["search_path"])
        template_path = os.path.join(val_dataset.root_dir, item["template_path"])

        sample = fo.Sample(filepath=search_path)
        sample["template_path"] = template_path

        if item.get("background", False):
            sample["ground_truth"] = fo.Keypoints() # Empty label
            sample.tags.append("background")
        else:
            # Store GT centroid as a Keypoint
            # We must normalize coords to [0, 1] for FiftyOne
            xg, yg = item["centroid"]
            xg_rel = xg / img_size
            yg_rel = yg / img_size
            
            gt_keypoint = fo.Keypoint(label="gt", points=[(xg_rel, yg_rel)])
            sample["ground_truth"] = fo.Keypoints(keypoints=[gt_keypoint])
            
            # Store raw GT heatmap
            gt_heatmap = val_dataset.generate_heatmap(item["centroid"])
            sample["gt_heatmap"] = fo.Heatmap(map=gt_heatmap.squeeze().numpy())

        samples_to_add.append(sample)

    dataset.add_samples(samples_to_add)

    # Run Model and Add Predictions
    print("Running inference and adding predictions to FiftyOne...")
    
    # Get an ordered list of sample IDs to update
    sample_ids = dataset.values("id")
    samples_to_save = []
    
    with torch.no_grad():
        idx_counter = 0
        for templates, searches, heatmaps in tqdm(val_loader, desc="Evaluating"):
            templates = templates.to(DEVICE)
            searches = searches.to(DEVICE)
            
            # Run model
            preds_logits = model(templates, searches)
            preds = torch.sigmoid(preds_logits)

            # Interpolate (as in your original code)
            preds = F.interpolate(preds, size=(img_size, img_size), mode='bilinear', align_corners=False)
            heatmaps = F.interpolate(heatmaps, size=(img_size, img_size), mode='bilinear', align_corners=False)

            # Get centroids
            centroids_pred = get_centroids_per_sample(preds)
            centroids_gt = get_centroids_per_sample(heatmaps) # From GT heatmap

            for i in range(len(templates)):
                # Get the corresponding FiftyOne sample
                sample_id = sample_ids[idx_counter]
                sample = dataset[sample_id]
                idx_counter += 1

                p = centroids_pred[i]
                t = centroids_gt[i]

                # Save predicted heatmap
                pred_hm_np = preds[i, 0].cpu().numpy()
                sample["pred_heatmap"] = fo.Heatmap(map=pred_hm_np)

                pred_keypoints = []
                if p is not None:
                    xp, yp, confidence = p
                    xp, yp = xp.item(), yp.item()
                    confidence = confidence.item()
                    
                    # Normalize for FiftyOne
                    xp_rel, yp_rel = xp / img_size, yp / img_size
                    
                    pred_keypoint = fo.Keypoint(
                        label="pred", 
                        points=[(xp_rel, yp_rel)], 
                        confidence=[confidence]
                    )
                    pred_keypoints.append(pred_keypoint)
                    sample["confidence"] = confidence
                
                # Add prediction keypoints (even if empty)
                sample["prediction"] = fo.Keypoints(keypoints=pred_keypoints)

                # Calculate and save distance error
                if p is not None and t is not None:
                    xp, yp, _ = p
                    xg, yg, _ = t
                    dist = np.sqrt((xp.item() - xg.item())**2 + (yp.item() - yg.item())**2)
                    sample["distance_error"] = dist
                else:
                    sample["distance_error"] = None # Handle misses

                samples_to_save.append(sample)

    # Save all updates to the dataset
    dataset.add_samples(samples_to_save)
    print(f"Successfully added predictions to {len(samples_to_save)} samples.")

    # Launch the App
    session = fo.launch_app(dataset)
    print("FiftyOne App launched. Press Ctrl+C in this terminal to close.")
    session.wait()

if __name__ == "__main__":
    main()