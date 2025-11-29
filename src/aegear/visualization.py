"""Visualization utilities for model inspection using FiftyOne."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import fiftyone as fo

from aegear.nn.training import get_centroids_per_sample


class FiftyOneDatasetBuilder:
    """Base class for building FiftyOne datasets from model predictions."""
    
    def __init__(self, dataset, model, device, img_size):
        self.dataset = dataset
        self.model = model
        self.device = device
        self.img_size = img_size
        
    def build_dataset(self, fo_dataset_name, batch_size=128, num_workers=4):
        """Build and populate a FiftyOne dataset.
        
        Parameters
        ----------
        fo_dataset_name : str
            Name for the FiftyOne dataset
        batch_size : int
            Batch size for inference
        num_workers : int
            Number of workers for data loading
            
        Returns
        -------
        fo.Dataset
            Populated FiftyOne dataset
        """
        raise NotImplementedError("Subclasses must implement build_dataset")
    
    def _create_dataloader(self, batch_size, num_workers):
        """Create a DataLoader with shuffle=False for consistent ordering."""
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )


class TrackingDatasetBuilder(FiftyOneDatasetBuilder):
    """Builder for tracking model evaluation datasets in FiftyOne."""
    
    def build_dataset(self, fo_dataset_name, batch_size=128, num_workers=4):
        """Build FiftyOne dataset for tracking model evaluation."""
        print("Populating FiftyOne dataset from metadata...")
        fo_dataset = fo.Dataset(fo_dataset_name, persistent=True, overwrite=True)
        
        # Add metadata samples
        samples_to_add = self._create_metadata_samples()
        fo_dataset.add_samples(samples_to_add)
        
        # Run inference and add predictions
        print("Running inference and adding predictions to FiftyOne...")
        self._add_predictions(fo_dataset, batch_size, num_workers)
        
        return fo_dataset
    
    def _create_metadata_samples(self):
        """Create FiftyOne samples from dataset metadata."""
        samples = []
        for item in tqdm(self.dataset.metadata, desc="Loading metadata"):
            search_path = os.path.join(self.dataset.root_dir, item["search_path"])
            template_path = os.path.join(self.dataset.root_dir, item["template_path"])
            
            sample = fo.Sample(filepath=search_path)
            
            # Store template as a custom field with its own visualizations
            sample["template_filepath"] = template_path
            
            # Add template image as embedded visualization if available
            # This allows viewing template alongside search image
            try:
                import PIL.Image as Image
                template_img = Image.open(template_path)
                sample["template_image"] = fo.Image(filepath=template_path)
            except:
                pass  # Skip if image loading fails
            
            if item.get("background", False):
                sample["ground_truth"] = fo.Keypoints()
                sample.tags.append("background")
            else:
                # Store GT centroid as keypoint
                xg, yg = item["centroid"]
                xg_rel = xg / self.img_size
                yg_rel = yg / self.img_size
                
                gt_keypoint = fo.Keypoint(label="gt", points=[(xg_rel, yg_rel)])
                sample["ground_truth"] = fo.Keypoints(keypoints=[gt_keypoint])
                
                # Store raw GT heatmap
                gt_heatmap = self.dataset.generate_heatmap(item["centroid"])
                sample["gt_heatmap"] = fo.Heatmap(map=gt_heatmap.squeeze().numpy())
            
            # Store template and search ROI information if available
            if "template_bbox" in item:
                bbox = item["template_bbox"]
                # Normalize bbox to [0, 1] for search image
                x, y, w, h = bbox
                sample["template_roi"] = fo.Detection(
                    label="template_roi",
                    bounding_box=[x/self.img_size, y/self.img_size, w/self.img_size, h/self.img_size]
                )
            
            # Store search ROI information if available (the region being searched)
            if "search_bbox" in item:
                bbox = item["search_bbox"]
                x, y, w, h = bbox
                sample["search_roi"] = fo.Detection(
                    label="search_roi",
                    bounding_box=[x/self.img_size, y/self.img_size, w/self.img_size, h/self.img_size]
                )
            
            # Store metadata for reference
            sample["sample_metadata"] = {
                "search_path": item["search_path"],
                "template_path": item["template_path"],
                "is_background": item.get("background", False)
            }
            
            samples.append(sample)
        
        return samples
    
    def _add_predictions(self, fo_dataset, batch_size, num_workers):
        """Run inference and add predictions to FiftyOne dataset."""
        loader = self._create_dataloader(batch_size, num_workers)
        sample_ids = fo_dataset.values("id")
        samples_to_save = []
        
        self.model.eval()
        with torch.no_grad():
            idx_counter = 0
            for templates, searches, heatmaps in tqdm(loader, desc="Evaluating"):
                templates = templates.to(self.device)
                searches = searches.to(self.device)
                
                # Run model
                preds_logits = self.model(templates, searches)
                preds = torch.sigmoid(preds_logits)
                
                # Interpolate
                preds = F.interpolate(preds, size=(self.img_size, self.img_size), 
                                    mode='bilinear', align_corners=False)
                heatmaps = F.interpolate(heatmaps, size=(self.img_size, self.img_size),
                                       mode='bilinear', align_corners=False)
                
                # Get centroids
                centroids_pred = get_centroids_per_sample(preds)
                centroids_gt = get_centroids_per_sample(heatmaps)
                
                for i in range(len(templates)):
                    sample_id = sample_ids[idx_counter]
                    sample = fo_dataset[sample_id]
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
                        xp_rel, yp_rel = xp / self.img_size, yp / self.img_size
                        
                        pred_keypoint = fo.Keypoint(
                            label="pred",
                            points=[(xp_rel, yp_rel)],
                            confidence=[confidence]
                        )
                        pred_keypoints.append(pred_keypoint)
                        sample["confidence"] = confidence
                    
                    sample["prediction"] = fo.Keypoints(keypoints=pred_keypoints)
                    
                    # Calculate distance error
                    if p is not None and t is not None:
                        xp, yp, _ = p
                        xg, yg, _ = t
                        dist = np.sqrt((xp.item() - xg.item())**2 + (yp.item() - yg.item())**2)
                        sample["distance_error"] = dist
                    else:
                        sample["distance_error"] = None
                    
                    samples_to_save.append(sample)
        
        fo_dataset.add_samples(samples_to_save)
        print(f"Successfully added predictions to {len(samples_to_save)} samples.")


class DetectionDatasetBuilder(FiftyOneDatasetBuilder):
    """Builder for detection model evaluation datasets in FiftyOne."""
    
    def build_dataset(self, fo_dataset_name, batch_size=128, num_workers=4):
        """Build FiftyOne dataset for detection model evaluation."""
        print("Populating FiftyOne dataset from metadata...")
        fo_dataset = fo.Dataset(fo_dataset_name, persistent=True, overwrite=True)
        
        # Add metadata samples
        samples_to_add = self._create_metadata_samples()
        fo_dataset.add_samples(samples_to_add)
        
        # Run inference and add predictions
        print("Running inference and adding predictions to FiftyOne...")
        self._add_predictions(fo_dataset, batch_size, num_workers)
        
        return fo_dataset
    
    def _create_metadata_samples(self):
        """Create FiftyOne samples from dataset metadata."""
        samples = []
        for item in tqdm(self.dataset.metadata, desc="Loading metadata"):
            image_path = os.path.join(self.dataset.root_dir, item["image_path"])
            
            sample = fo.Sample(filepath=image_path)
            
            if item.get("background", False):
                sample["ground_truth"] = fo.Keypoints()
                sample.tags.append("background")
            else:
                # Store GT centroid as keypoint
                xg, yg = item["centroid"]
                xg_rel = xg / self.img_size
                yg_rel = yg / self.img_size
                
                gt_keypoint = fo.Keypoint(label="gt", points=[(xg_rel, yg_rel)])
                sample["ground_truth"] = fo.Keypoints(keypoints=[gt_keypoint])
                
                # Store raw GT heatmap
                gt_heatmap = self.dataset.generate_heatmap(item["centroid"])
                sample["gt_heatmap"] = fo.Heatmap(map=gt_heatmap.squeeze().numpy())
            
            samples.append(sample)
        
        return samples
    
    def _add_predictions(self, fo_dataset, batch_size, num_workers):
        """Run inference and add predictions to FiftyOne dataset."""
        loader = self._create_dataloader(batch_size, num_workers)
        sample_ids = fo_dataset.values("id")
        samples_to_save = []
        
        self.model.eval()
        with torch.no_grad():
            idx_counter = 0
            for images, heatmaps in tqdm(loader, desc="Evaluating"):
                images = images.to(self.device)
                
                # Run model
                preds_logits = self.model(images)
                preds = torch.sigmoid(preds_logits)
                
                # Interpolate
                preds = F.interpolate(preds, size=(self.img_size, self.img_size),
                                    mode='bilinear', align_corners=False)
                heatmaps = F.interpolate(heatmaps, size=(self.img_size, self.img_size),
                                       mode='bilinear', align_corners=False)
                
                # Get centroids
                centroids_pred = get_centroids_per_sample(preds)
                centroids_gt = get_centroids_per_sample(heatmaps)
                
                for i in range(len(images)):
                    sample_id = sample_ids[idx_counter]
                    sample = fo_dataset[sample_id]
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
                        xp_rel, yp_rel = xp / self.img_size, yp / self.img_size
                        
                        pred_keypoint = fo.Keypoint(
                            label="pred",
                            points=[(xp_rel, yp_rel)],
                            confidence=[confidence]
                        )
                        pred_keypoints.append(pred_keypoint)
                        sample["confidence"] = confidence
                    
                    sample["prediction"] = fo.Keypoints(keypoints=pred_keypoints)
                    
                    # Calculate distance error
                    if p is not None and t is not None:
                        xp, yp, _ = p
                        xg, yg, _ = t
                        dist = np.sqrt((xp.item() - xg.item())**2 + (yp.item() - yg.item())**2)
                        sample["distance_error"] = dist
                    else:
                        sample["distance_error"] = None
                    
                    samples_to_save.append(sample)
        
        fo_dataset.add_samples(samples_to_save)
        print(f"Successfully added predictions to {len(samples_to_save)} samples.")


def launch_fiftyone_app(dataset):
    """Launch the FiftyOne app for dataset visualization.
    
    Parameters
    ----------
    dataset : fo.Dataset
        FiftyOne dataset to visualize
        
    Returns
    -------
    fo.Session
        FiftyOne session object
    """
    session = fo.launch_app(dataset)
    print("FiftyOne App launched. Press Ctrl+C in this terminal to close.")
    return session