"""
Module containing various training-related utilities and functions.
"""

import os
import json
import logging
import time
import sys
from tqdm import tqdm

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms.functional as TF

from aegear.nn.datasets import CachedTrackingDataset
from aegear.nn.model import EfficientUNet
from aegear.utils import get_latest_model_path


BATCH_UPDATE_FREQUENCY = 100  # Log every N batches when not using progress bar

def setup_logging(log_level=logging.INFO):
    """Set up logging for training.
    
    Args:
        log_level (int): Logging level.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=log_level
    )
    logger = logging.getLogger("aegear.train")
    return logger


def _should_disable_progress() -> bool:
    """Return True when tqdm progress bars should be suppressed."""
    if os.getenv("RUNPOD_POD_ID"):
        return True
    if os.getenv("CLEARML_TASK") or os.getenv("CLEARML_TASK_ID"):
        return True
    return not sys.stdout.isatty()

def get_device():
    """Get the best available torch device (MPS, CUDA, or CPU).
    
    Returns:
        torch.device: The selected device.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_datasets(cache_dir, datasets, batch_size=128, gaussian_sigma=15.0):
    """Load training and validation datasets and create DataLoaders.
    
    Args:
        cache_dir (str): Directory containing cached datasets.
        datasets (list): List of dataset names.
        batch_size (int): Batch size for DataLoader.
        gaussian_sigma (float): Sigma for Gaussian heatmap.
    
    Returns:
        tuple: (train_loader, val_loader, train_dataset, val_dataset)
    """
    train_dataset = ConcatDataset([
        CachedTrackingDataset(os.path.join(cache_dir, name, "train"), gaussian_sigma=gaussian_sigma)
        for name in datasets
    ])
    val_dataset = ConcatDataset([
        CachedTrackingDataset(os.path.join(cache_dir, name, "val"), gaussian_sigma=gaussian_sigma)
        for name in datasets
    ])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, train_dataset, val_dataset

def setup_model(weights="IMAGENET1K_V1", continue_training=False, use_best_model=False, model_dir="../data/training/models/efficient_unet", pretrained_model_dir="../models/", device=None, **model_kwargs):
    """Set up the EfficientUNet model for training or fine-tuning.
    
    Args:
        weights (str): Pretrained weights identifier.
        continue_training (bool): Whether to continue training from a checkpoint.
        use_best_model (bool): Use the best model checkpoint.
        model_dir (str): Directory for saving/loading models.
        pretrained_model_dir (str): Directory for pretrained models.
        device (torch.device): Device to load model on.
        **model_kwargs: Additional model arguments.
    
    Returns:
        EfficientUNet: Initialized model.
    """
    model = EfficientUNet(weights=weights, **model_kwargs)
    logger = logging.getLogger("aegear.train")
    if continue_training:
        if use_best_model:
            best_model_path = os.path.join(model_dir, "best_model.pth")
            assert os.path.exists(best_model_path)
        else:
            unet_model_filename = "model_efficient_unet"
            best_model_path = get_latest_model_path(pretrained_model_dir, unet_model_filename)
        logger.info(f"Continuing training of the UNet model from: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
    else:
        logger.info("Training a new UNet model from ImageNet weights.")
    model.to(device)
    return model

def freeze_model_layers(model, freeze_layers):
    """Freeze specified layers in the model (set requires_grad=False).
    
    Args:
        model: Model instance.
        freeze_layers (list): List of layers (or names) to freeze.
    """
    for param in model.parameters():
        param.requires_grad = True
    for layer in freeze_layers:
        # If layer is a string, resolve to model attribute
        if isinstance(layer, str):
            resolved_layer = model
            for attr in layer.split('.'):
                resolved_layer = getattr(resolved_layer, attr)
            layer_obj = resolved_layer
        else:
            layer_obj = layer

        for param in layer_obj.parameters():
            param.requires_grad = False

def set_layers_eval(model, layers):
    """Set specified layers to evaluation mode.
    
    Args:
        model: Model instance.
        layers (list): List of layers (or names) to set to eval mode.
    """
    for layer in layers:
        # If layer is a string, resolve to model attribute
        if isinstance(layer, str):
            resolved_layer = model
            for attr in layer.split('.'):
                resolved_layer = getattr(resolved_layer, attr)
            layer_obj = resolved_layer
        else:
            layer_obj = layer
        layer_obj.eval()

def load_training_stages_from_config(model, stages_config):
    """
    Load training stages from a dict/list structure (e.g., from YAML) and resolve layer names.
    
    Args:
        model: The model instance (EfficientUNet or SiameseTracker).
        stages_config: List of stage dicts or dict with stages key.
    
    Returns:
        List of training stage dicts with freeze_layers resolved to actual layer objects.
    
    Example:
        stages_config = [
            {
                "name": "Stage 1",
                "freeze_layers": ["enc1", "enc2", "enc3", "enc4"],
                "epochs": 10,
                "lr": 0.001
            }
        ]
    """
    # Handle both direct list and dict with 'stages' key
    if isinstance(stages_config, dict) and 'stages' in stages_config:
        training_stages = stages_config['stages']
    elif isinstance(stages_config, list):
        training_stages = stages_config
    else:
        raise ValueError("stages_config must be a list of stages or dict with 'stages' key")
    
    # Resolve layer names to actual model layer objects
    resolved_stages = []
    for stage in training_stages:
        resolved_stage = stage.copy()
        if 'freeze_layers' in stage:
            resolved_layers = []
            for layer_name in stage['freeze_layers']:
                layer = model
                for attr in layer_name.split('.'):
                    layer = getattr(layer, attr)
                resolved_layers.append(layer)
            resolved_stage['freeze_layers'] = resolved_layers
        resolved_stages.append(resolved_stage)
    
    return resolved_stages


def load_training_stages(model, stages_path=None):
    """
    Load training stages from a JSON file and resolve layer names to model attributes.
    
    Args:
        model: The model instance (EfficientUNet or SiameseTracker).
        stages_path: Path to the JSON file.
    
    Returns:
        List of training stage dicts with freeze_layers resolved.
    """
    if not os.path.exists(stages_path):
        raise IOError(f"Training stages file not found: {stages_path}")


    with open(stages_path, 'r') as f:
        raw = f.read()
        print("[DEBUG] Raw training stages file contents:")
        print(raw)
        training_stages = json.loads(raw)

    # Use the new function to handle resolution
    return load_training_stages_from_config(model, training_stages)

def get_default_training_stages(model_name: str, epochs: int = 10, lr: float = 1e-4):
    """
    Return default training stages for the given model name ('efficient_unet' or 'siamese').
    The returned format matches what load_training_stages expects (layer names as strings).
    Args:
        model_name: 'efficient_unet' or 'siamese'
        epochs: Number of epochs for the stage(s).
        lr: Learning rate for the stage(s).
    Returns:
        List of training stage dicts with freeze_layers as strings.
    """
    if model_name == "efficient_unet":
        return [
            {
                "freeze_layers": ["enc1", "enc2", "enc3", "enc4"],
                "epochs": epochs,
                "lr": lr,
            }
        ]
    elif model_name == "siamese":
        # Always use positive integer epochs for both stages
        stage1_epochs = max(1, int(epochs))
        stage2_epochs = max(1, int(epochs // 2))
        return [
            {
                "freeze_layers": ["enc1", "enc2", "enc3", "enc4", "enc5"],
                "epochs": stage1_epochs,
                "lr": 5.0 * lr,
            },
            {
                "freeze_layers": ["enc1", "enc2", "enc3", "enc4"],
                "epochs": stage2_epochs,
                "lr": lr,
            }
        ]
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def collect_val_results(val_batches, device):
    """
    Collect validation results from batches for visualization and metrics.

    Args:
        val_batches (list): List of validation batches.
        device (torch.device): Device for tensor operations.

    Returns:
        list: List of dicts containing results per sample.

    Raises:
        ValueError: If a batch does not have 3 or 4 elements (unexpected batch size).
    """
    val_results = []
    # Generalized for both EfficientUNet and SiameseTracker
    for batch in val_batches:
        if len(batch) == 3:
            search, target, output = batch
            template = None
        elif len(batch) == 4:
            template, search, target, output = batch
        else:
            raise ValueError(f"Unexpected batch size {len(batch)} in validation results. Expected 3 or 4.")

        pred_resized = F.interpolate(torch.sigmoid(output), size=search.shape[-2:], mode='bilinear', align_corners=False)
        target_resized = F.interpolate(target, size=search.shape[-2:], mode='bilinear', align_corners=False)
        centroids_pred = get_centroids_per_sample(pred_resized)
        centroids_gt = get_centroids_per_sample(target_resized)

        for i in range(search.size(0)):
            p = centroids_pred[i]
            t = centroids_gt[i]
            if p is None or t is None:
                continue
            x_pred, y_pred, confidence = p
            x_gt, y_gt, _ = t
            xp, yp = x_pred.item(), y_pred.item()
            xg, yg = x_gt.item(), y_gt.item()
            dist = np.sqrt((xp - xg) ** 2 + (yp - yg) ** 2)
            result = {
                'search': search[i].cpu(),
                'gt_heatmap': target_resized[i, 0].cpu(),
                'pred_heatmap': pred_resized[i, 0].cpu(),
                'gt_centroid': (xg, yg),
                'pred_centroid': (xp, yp),
                'confidence': confidence.item(),
                'distance': dist,
            }
            if template is not None:
                result['template'] = template[i].cpu()
            val_results.append(result)
    return val_results

def get_model_type(model, explicit_type=None):
    """
    Determine the model type ('efficient_unet' or 'siamese') from the model instance or explicit argument.
    
    Args:
        model: Model instance.
        explicit_type (str, optional): Explicit model type if provided.
    
    Returns:
        str: Model type ('efficient_unet' or 'siamese').
    """
    if explicit_type:
        return explicit_type
    name = model.__class__.__name__.lower()
    if name.startswith("efficientunet"):
        return "efficient_unet"
    elif name.startswith("siamesetracker"):
        return "siamese"
    raise ValueError("Unknown model type for training.")

def process_train_batch(model, batch, model_type, device, loss_fn, return_components=False):
    """
    Process a single training batch for the given model type.
    
    Args:
        model: Model instance.
        batch: Batch data from DataLoader.
        model_type (str): Model type ('efficient_unet' or 'siamese').
        device: Torch device.
        loss_fn: Loss function.
        return_components (bool): If True, return loss components.
    
    Returns:
        tuple: (loss, output) or (loss, output, components) if return_components=True
    """
    if model_type == "efficient_unet":
        if len(batch) == 2:
            search, target = batch
            output = model(search.to(device))
            if return_components:
                loss, components = loss_fn(output, target.to(device), return_components=True)
                return loss, output, components
            else:
                loss = loss_fn(output, target.to(device))
        else:
            _, search, target = batch
            output = model(search.to(device))
            if return_components:
                loss, components = loss_fn(output, target.to(device), return_components=True)
                return loss, output, components
            else:
                loss = loss_fn(output, target.to(device))
    elif model_type == "siamese":
        template, search, target = batch
        output = model(template.to(device), search.to(device))
        if return_components:
            loss, components = loss_fn(output, target.to(device), template.to(device), search.to(device), return_components=True)
            return loss, output, components
        else:
            loss = loss_fn(output, target.to(device), template.to(device), search.to(device))
    else:
        raise ValueError("Unknown model_type in training loop.")
    return loss, output

def process_val_batch(model, batch, model_type, device, loss_fn, return_components=False):
    """
    Process a single validation batch for the given model type.
    
    Args:
        model: Model instance.
        batch: Batch data from DataLoader.
        model_type (str): Model type ('efficient_unet' or 'siamese').
        device: Torch device.
        loss_fn: Loss function.
        return_components (bool): If True, return loss components.
    
    Returns:
        tuple: (loss, batch_tuple) or (loss, batch_tuple, components) if return_components=True
    """
    if model_type == "efficient_unet":
        if len(batch) == 2:
            search, target = batch
            output = model(search.to(device))
            if return_components:
                loss, components = loss_fn(output, target.to(device), return_components=True)
                return loss, (search, target, output), components
            else:
                loss = loss_fn(output, target.to(device))
                return loss, (search, target, output)
        else:
            _, search, target = batch
            output = model(search.to(device))
            if return_components:
                loss, components = loss_fn(output, target.to(device), return_components=True)
                return loss, (search, target, output), components
            else:
                loss = loss_fn(output, target.to(device))
                return loss, (search, target, output)
    elif model_type == "siamese":
        template, search, target = batch
        output = model(template.to(device), search.to(device))
        if return_components:
            loss, components = loss_fn(output, target.to(device), template.to(device), search.to(device), return_components=True)
            return loss, (template, search, target, output), components
        else:
            loss = loss_fn(output, target.to(device), template.to(device), search.to(device))
            return loss, (template, search, target, output)
    else:
        raise ValueError("Unknown model_type in validation loop.")

def get_visualizer(model_type, model, device, val_results, stage, epoch, output_dir):
    """
    Get the appropriate visualizer instance for the model type.
    
    Args:
        model_type (str): Model type ('efficient_unet' or 'siamese').
        model: Model instance.
        device: Torch device.
        val_results: Validation results.
        stage (int): Training stage index.
        epoch (int): Epoch index.
        output_dir (str): Directory for visualizer outputs.
    
    Returns:
        object or None: Visualizer instance or None if not applicable.
    """
    if model_type == "efficient_unet":
        return EfficientUNetVisualizer(model, device, val_results, stage, epoch, output_dir=output_dir)
    elif model_type == "siamese":
        return SiameseTrackingVisualizer(model, device, val_results, stage, epoch, output_dir=output_dir)
    return None

def create_scheduler(optimizer, scheduler_config, **kwargs):
    """
    Create a PyTorch LR scheduler from a config dict.
    Args:
        optimizer: Optimizer instance.
        scheduler_config (dict): Dict with 'type' and scheduler-specific kwargs.
        train_loader: DataLoader (needed for OneCycleLR).
        epochs: Number of epochs (needed for OneCycleLR).
    Returns:
        torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau
    """
    if scheduler_config is None:
        return None
    sched_type = scheduler_config.get('type', 'ReduceLROnPlateau')
    params = scheduler_config.get('params', {})

    # For OneCycleLR, require epochs as argument
    if sched_type == 'OneCycleLR':
        steps_per_epoch = kwargs.get('steps_per_epoch', None)
        epochs = kwargs.get('epochs', None)

        if steps_per_epoch is None or epochs is None:
            raise ValueError("steps_per_epoch and epochs must be provided for OneCycleLR scheduler.")

        return torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=steps_per_epoch, epochs=epochs, **params)
    elif sched_type == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    elif sched_type == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    elif sched_type == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")

def scheduler_config_to_json(scheduler_config):
    """Serialize scheduler config dict to JSON string."""
    return json.dumps(scheduler_config)

def scheduler_config_from_json(json_str):
    """Deserialize scheduler config dict from JSON string."""
    return json.loads(json_str)

def get_epoch_progress_message(current_epoch, total_epochs, epoch_time, epoch_times):
    """Generate a progress message for the current epoch."""
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    epochs_left = total_epochs - current_epoch
    eta = avg_epoch_time * epochs_left
    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
    epoch_time_str = time.strftime('%H:%M:%S', time.gmtime(epoch_time))

    return (f"Epoch {current_epoch}/{total_epochs} completed. "
        f"Time: {epoch_time_str}. "
        f"ETA: {eta_str}.")

def compute_validation_metrics(model, val_loader, device, model_type, loss_fn=None):
    """
    Compute validation metrics: average centroid distance, confidence, within-radius percentages, and loss components.
    
    Args:
        model: The trained model.
        val_loader: Validation data loader.
        device: Torch device.
        model_type (str): Model type ('efficient_unet' or 'siamese').
        loss_fn: Optional loss function to compute loss components.
    
    Returns:
        dict: Dictionary containing metrics and loss components.
    """
    model.eval()
    total_distances = []
    total_confidences = []
    within_radius = {3: 0, 5: 0, 10: 0}
    n_samples = 0
    loss_components_sum = {}
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(
            val_loader,
            desc="Computing validation metrics",
            leave=False,
            disable=_should_disable_progress()
        ):
            if model_type == "efficient_unet":
                if len(batch) == 2:
                    imgs, heatmaps = batch
                    imgs = imgs.to(device)
                    heatmaps = heatmaps.to(device)
                    preds = torch.sigmoid(model(imgs))
                else:
                    imgs, _, heatmaps = batch
                    imgs = imgs.to(device)
                    heatmaps = heatmaps.to(device)
                    preds = torch.sigmoid(model(imgs))
            elif model_type == "siamese":
                template, search, heatmaps = batch
                template = template.to(device)
                search = search.to(device)
                heatmaps = heatmaps.to(device)
                preds = torch.sigmoid(model(template, search))
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            centroids_pred = get_centroids_per_sample(preds)
            centroids_gt = get_centroids_per_sample(heatmaps)
            
            # Compute loss components if loss function is provided
            if loss_fn is not None:
                if model_type == "efficient_unet":
                    # Need logits for loss computation (recompute to get logits, not sigmoid)
                    logits = model(imgs)
                    _, components = loss_fn(logits, heatmaps, return_components=True)
                elif model_type == "siamese":
                    logits = model(template, search)
                    _, components = loss_fn(logits, heatmaps, template, search, return_components=True)
                
                # Accumulate components (only weighted versions and bce_loss/total_loss)
                for key, value in components.items():
                    if '_weighted' in key or key == 'total_loss' or key == 'bce_loss':
                        if key not in loss_components_sum:
                            loss_components_sum[key] = 0.0
                        loss_components_sum[key] += value
                n_batches += 1
            
            for i in range(len(imgs) if model_type == "efficient_unet" else len(template)):
                p = centroids_pred[i]
                t = centroids_gt[i]
                
                if p is None or t is None:
                    continue
                
                x_pred, y_pred, confidence = p
                x_gt, y_gt, _ = t
                
                xp, yp = x_pred.item(), y_pred.item()
                xg, yg = x_gt.item(), y_gt.item()
                confidence = confidence.item()
                
                dist = np.sqrt((xp - xg) ** 2 + (yp - yg) ** 2)
                total_distances.append(dist)
                total_confidences.append(confidence)
                
                for r in within_radius:
                    if dist <= r:
                        within_radius[r] += 1
                n_samples += 1
    
    if n_samples == 0:
        return {
            'avg_distance': 0.0,
            'avg_confidence': 0.0,
            'within_3px': 0.0,
            'within_5px': 0.0,
            'within_10px': 0.0,
            'n_samples': 0
        }
    
    metrics = {
        'avg_distance': np.mean(total_distances),
        'avg_confidence': np.mean(total_confidences),
        'within_3px': within_radius[3] / n_samples,
        'within_5px': within_radius[5] / n_samples,
        'within_10px': within_radius[10] / n_samples,
        'n_samples': n_samples
    }
    
    # Add averaged loss components to metrics
    if loss_components_sum and n_batches > 0:
        for key, value in loss_components_sum.items():
            metrics[f'loss/{key}'] = value / n_batches
    
    return metrics

def save_model_with_clearml(model, path, clearml_task=None, artifact_name=None, metadata=None):
    """
    Save a model checkpoint and register it with ClearML if a task is provided.

    Args:
        model: The PyTorch model to save.
        path (str): Path to save the model.
        clearml_task: ClearML Task object or None.
        artifact_name (str): Name for the artifact in ClearML.
        metadata (dict): Optional metadata to attach.
    """
    torch.save(model.state_dict(), path)
    if clearml_task is not None:
        name = artifact_name if artifact_name else os.path.basename(path)
        clearml_task.upload_artifact(
            name=name,
            artifact_object=path,
            metadata=metadata
        )

def train(
    model,
    train_loader,
    val_loader,
    device,
    model_dir,
    checkpoint_dir,
    epoch_vis,
    training_stages,
    loss_fn=None,
    epoch_save_interval=1,
    model_type=None,
    use_visualizer=False,
    weight_decay=5e-3,
    clearml_task=None,
    scheduler_config=None,
):
    """
    Unified training function for EfficientUNet and SiameseTracker.
    model_type: 'efficient_unet' or 'siamese'. If None, inferred from model class name.
    """
    # Ensure model_dir and checkpoint_dir exist
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Log configuration before training starts
    config_dict = {
        'model_type': get_model_type(model, model_type),
        'model_dir': model_dir,
        'checkpoint_dir': checkpoint_dir,
        'epoch_vis': epoch_vis,
        'epoch_save_interval': epoch_save_interval,
        'weight_decay': weight_decay,
        'use_visualizer': use_visualizer,
        'scheduler_config': scheduler_config,
        'loss_fn': loss_fn.__class__.__name__ if loss_fn is not None else None,
    }
    if clearml_task is not None:
        logger = clearml_task.get_logger()
        logger.report_text(f"Training configuration:\n{json.dumps(config_dict, indent=2)}", logging.INFO, iteration=0)
    else:
        logging.getLogger("aegear.train").info(f"Training configuration:\n{json.dumps(config_dict, indent=2)}")

    best_val_loss = float('inf')
    losses = []
    model_type = get_model_type(model, model_type)

    suppress_progress = (clearml_task is not None) or _should_disable_progress()

    try:
        total_train_batches = len(train_loader)
    except TypeError:
        total_train_batches = None
    try:
        total_val_batches = len(val_loader)
    except TypeError:
        total_val_batches = None

    # Number of iteration (epoch) across all stages
    global_iteration = 0

    for stage, training_stage in enumerate(training_stages):
        freeze_layers = training_stage["freeze_layers"]
        epochs = training_stage["epochs"]
        freeze_model_layers(model, freeze_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_stage["lr"], weight_decay=weight_decay)
        # Use scheduler_config if provided, else default ReduceLROnPlateau
        scfg = scheduler_config if scheduler_config is not None else {
            'type': 'ReduceLROnPlateau',
            'params': {'mode': 'min', 'factor': 0.5, 'patience': 3}
        }
        # For OneCycleLR, pass epochs as kwarg
        if scfg.get('type') == 'OneCycleLR':
            scheduler = create_scheduler(optimizer, scfg, steps_per_epoch=len(train_loader), epochs=epochs)
        else:
            scheduler = create_scheduler(optimizer, scfg)

        epoch_times = []
        total_epochs = epochs
        
        for epoch in range(epochs):

            # Increment global iteration
            global_iteration += 1

            epoch_start = time.time()
            model.train()
            set_layers_eval(model, freeze_layers)
            train_loss = 0.0
            train_loss_components = {}
            if suppress_progress:
                logging.getLogger("aegear.train").info(
                    f"Stage {stage + 1}, Training epoch {epoch + 1}/{epochs}"
                )
            train_bar = tqdm(
                train_loader,
                desc=f"Stage {stage + 1}, Training {epoch + 1}",
                leave=False,
                disable=suppress_progress
            )
            for batch_idx, batch in enumerate(train_bar, start=1):
                result = process_train_batch(model, batch, model_type, device, loss_fn, return_components=(clearml_task is not None))
                if clearml_task is not None:
                    loss, _, components = result
                    # Accumulate components
                    for key, value in components.items():
                        if key not in train_loss_components:
                            train_loss_components[key] = 0.0
                        train_loss_components[key] += value
                else:
                    loss, _ = result
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Step OneCycleLR scheduler per batch
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

                train_loss += loss.item()
                if not suppress_progress:
                    train_bar.set_postfix(loss=loss.item())
                elif total_train_batches and batch_idx % BATCH_UPDATE_FREQUENCY == 0:
                    logging.getLogger("aegear.train").info(
                        f"Stage {stage + 1} epoch {epoch + 1}: batch {batch_idx}/{total_train_batches} loss={loss.item():.4f}"
                    )

            model.eval()
            val_loss = 0.0
            val_loss_components = {}
            val_batches = []
            if suppress_progress:
                logging.getLogger("aegear.train").info(
                    f"Stage {stage + 1}, Validation epoch {epoch + 1}/{epochs}"
                )
            val_bar = tqdm(
                val_loader,
                desc=f"Stage {stage + 1}, Validation {epoch + 1}",
                leave=False,
                disable=suppress_progress
            )
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_bar, start=1):
                    result = process_val_batch(model, batch, model_type, device, loss_fn, return_components=(clearml_task is not None))
                    if clearml_task is not None:
                        loss, val_batch, components = result
                        # Accumulate components
                        for key, value in components.items():
                            if key not in val_loss_components:
                                val_loss_components[key] = 0.0
                            val_loss_components[key] += value
                    else:
                        loss, val_batch = result
                    
                    # Detach all tensors in val_batch before storing
                    if isinstance(val_batch, tuple):
                        detached_batch = tuple(x.detach().cpu() if torch.is_tensor(x) else x for x in val_batch)
                    else:
                        detached_batch = val_batch.detach().cpu() if torch.is_tensor(val_batch) else val_batch
                    val_batches.append(detached_batch)
                    val_loss += loss.item()
                    if not suppress_progress:
                        val_bar.set_postfix(loss=loss.item())
                    elif total_val_batches and batch_idx % BATCH_UPDATE_FREQUENCY == 0:
                        logging.getLogger("aegear.train").info(
                            f"Stage {stage + 1} validation epoch {epoch + 1}: batch {batch_idx}/{total_val_batches} loss={loss.item():.4f}"
                        )

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_results = collect_val_results(val_batches, device)
            losses.append((train_loss, val_loss))
            # Step scheduler depending on type (OneCycleLR is stepped per batch, not per epoch)
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                elif not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

            # Logging to ClearML if available
            if clearml_task is not None:
                logger = clearml_task.get_logger()
                logger.report_scalar("loss", "train", iteration=global_iteration, value=train_loss)
                logger.report_scalar("loss", "val", iteration=global_iteration, value=val_loss)
                
                # Log individual loss components
                if train_loss_components:
                    num_train_batches = len(train_loader)
                    for key, value in train_loss_components.items():
                        avg_value = value / num_train_batches
                        logger.report_scalar(f"loss_components/train", key, iteration=global_iteration, value=avg_value)
                
                if val_loss_components:
                    num_val_batches = len(val_loader)
                    for key, value in val_loss_components.items():
                        avg_value = value / num_val_batches
                        logger.report_scalar(f"loss_components/val", key, iteration=global_iteration, value=avg_value)

                # Log sample images from visualizer
                if use_visualizer:
                    visualizer = get_visualizer(model_type, model, device, val_results, stage, epoch, epoch_vis)
                    if visualizer:
                        perf_fig = visualizer.performance(num_samples=5, save=False)
                        act_fig = visualizer.activations(num_samples=3, save=False)
                        
                        logger.report_matplotlib_figure(
                            title="Sample Evaluation",
                            series=f"Epoch {epoch+1}, Stage {stage+1}",
                            iteration=global_iteration,
                            report_image=True,  # These are samples evaluated, so we report them like debug samples.
                            figure=perf_fig
                        )
                        plt.close(perf_fig)

                        logger.report_matplotlib_figure(
                            title="Activation Evaluation",
                            series=f"Epoch {epoch+1}, Stage {stage+1}",
                            iteration=global_iteration,
                            report_image=False,  # These are plots to be inspected (show up among 'Plots' in ClearML UI)
                            figure=act_fig
                        )
                        plt.close(act_fig)
            else:
                # Fallback to stdout logging and tqdm
                logging.getLogger("aegear.train").info(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f} | Val: {val_loss:.4f}")
                if use_visualizer:
                    visualizer = get_visualizer(model_type, model, device, val_results, stage, epoch, epoch_vis)
                    if visualizer:
                        visualizer.performance(num_samples=5)
                        visualizer.activations(num_samples=3)


            # --- Epoch timing and ETA logging ---
            epoch_end = time.time()
            epoch_time = epoch_end - epoch_start
            epoch_times.append(epoch_time)
            msg = get_epoch_progress_message(epoch+1, total_epochs, epoch_time, epoch_times)
            if clearml_task is not None:
                logger.report_text(msg, logging.INFO, iteration=global_iteration)
            else:
                logging.getLogger("aegear.train").info(msg)
            # --- End epoch timing and ETA logging ---

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logging.getLogger("aegear.train").info("New best model, saving.")
                save_model_with_clearml(
                    model,
                    f'{model_dir}/best_model.pth',
                    clearml_task,
                    artifact_name="best_model",
                    metadata={
                        "type": "best",
                        "stage": stage,
                        "epoch": epoch+1,
                        "val_loss": val_loss
                    }
                )
            if (epoch + 1) % epoch_save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_stage_{stage+1}_epoch_{epoch+1}.pth')
                save_model_with_clearml(
                    model,
                    checkpoint_path,
                    clearml_task,
                    artifact_name=f"checkpoint_stage_{stage+1}_epoch_{epoch+1}",
                    metadata={
                        "type": "checkpoint",
                        "stage": stage,
                        "epoch": epoch+1,
                        "val_loss": val_loss
                    }
                )

    # Compute final validation metrics on the best model
    logger_obj = logging.getLogger("aegear.train")
    logger_obj.info("Computing final validation metrics on best model...")
    
    # Load best model
    best_model_path = f'{model_dir}/best_model.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        
        # Compute metrics (pass loss_fn to get loss components)
        final_metrics = compute_validation_metrics(model, val_loader, device, model_type, loss_fn=loss_fn)
        
        # Log metrics
        logger_obj.info(f"\nFinal Validation Metrics:")
        logger_obj.info(f"  Average centroid distance: {final_metrics['avg_distance']:.2f} px")
        logger_obj.info(f"  Average confidence: {final_metrics['avg_confidence']:.4f}")
        logger_obj.info(f"  Within 3px: {final_metrics['within_3px']:.2%}")
        logger_obj.info(f"  Within 5px: {final_metrics['within_5px']:.2%}")
        logger_obj.info(f"  Within 10px: {final_metrics['within_10px']:.2%}")
        logger_obj.info(f"  Total samples: {final_metrics['n_samples']}")
        
        # Log loss components if available
        loss_component_keys = [k for k in final_metrics.keys() if k.startswith('loss/')]
        if loss_component_keys:
            logger_obj.info(f"\nFinal Loss Components:")
            for key in sorted(loss_component_keys):
                clean_name = key.replace('loss/', '')
                logger_obj.info(f"  {clean_name}: {final_metrics[key]:.6f}")
        
        # Log to ClearML if available
        if clearml_task is not None:
            logger = clearml_task.get_logger()
            logger.report_single_value("final_avg_distance", final_metrics['avg_distance'])
            logger.report_single_value("final_avg_confidence", final_metrics['avg_confidence'])
            logger.report_single_value("final_within_3px", final_metrics['within_3px'])
            logger.report_single_value("final_within_5px", final_metrics['within_5px'])
            logger.report_single_value("final_within_10px", final_metrics['within_10px'])
            logger.report_single_value("final_n_samples", final_metrics['n_samples'])
            
            # Log final loss components to ClearML Summary
            for key in loss_component_keys:
                clean_name = key.replace('loss/', '')
                logger.report_single_value(f"final_loss_{clean_name}", final_metrics[key])
    else:
        logger_obj.warning(f"Best model not found at {best_model_path}, skipping final metrics computation.")

    return losses


def get_confidence(heatmap):
    """Get confidence score from a heatmap by finding the maximum value.
    
    Args:
        heatmap (torch.Tensor): Heatmap tensor of shape (B, 1, H, W).
    
    Returns:
        float: Confidence score (max value in heatmap).
    """
    b, _, _, w = heatmap.shape
    flat_idx = torch.argmax(heatmap.view(b, -1), dim=1)
    y = flat_idx // w
    x = flat_idx % w
    return heatmap[0, 0, y, x].item()


def overlay_heatmap_on_rgb(rgb_tensor, heatmap, alpha=0.5, centroid_color=(0, 1, 0)):
    """Overlay heatmap onto RGB image and draw a circle at the predicted centroid.
    
    Args:
        rgb_tensor (torch.Tensor): RGB image tensor of shape (3, H, W).
        heatmap (np.ndarray): Heatmap array of shape (H, W).
        alpha (float): Blending weight for overlay.
        centroid_color (tuple): (R, G, B) color for centroid (0-1 range).
    
    Returns:
        np.ndarray: Overlay image of shape (H, W, 3).
    """
    rgb = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    rgb = rgb * 0.229 + 0.485
    rgb = rgb.clip(0, 1)

    heatmap_color = plt.cm.hot(heatmap)[..., :3]
    overlay = (1 - alpha) * rgb + alpha * heatmap_color

    # Find centroid
    flat_idx = heatmap.reshape(-1).argmax()
    h, w = heatmap.shape
    cy = flat_idx // w
    cx = flat_idx % w

    # Draw circle
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    cx_int, cy_int = int(cx), int(cy)
    color_bgr = tuple(int(c * 255) for c in reversed(centroid_color))
    cv2.circle(overlay_uint8, (cx_int, cy_int), 4, color_bgr, thickness=1)

    return overlay_uint8 / 255.0


def denormalize(img_tensor, clamp=True):
    """Denormalize an image tensor using ImageNet mean and std.
    
    Args:
        img_tensor (torch.Tensor): Normalized image tensor.
        clamp (bool): Whether to clamp output to [0, 1].
    
    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    out = img_tensor * std + mean
    return out.clamp(0, 1) if clamp else out


def get_centroids_per_sample(heatmap):
    """Get centroids from a batch of heatmaps.
    
    Args:
        heatmap (torch.Tensor): Batch of heatmaps (B, 1, H, W).
    
    Returns:
        list: List of (x, y, confidence) tuples or None per sample.
    """
    b, _, _, w = heatmap.shape
    heatmaps = heatmap.squeeze(1)
    centroids = []
    for i in range(b):
        hm = heatmaps[i]
        hm_sum = hm.mean().item()
        if hm_sum < 1e-8:
            centroids.append(None)
        else:
            flat_idx = torch.argmax(hm)
            y = flat_idx // w
            x = flat_idx % w
            conf = hm[y, x]
            centroids.append((x.float(), y.float(), conf.float()))
    return centroids


class WeightedBCEWithLogitsLoss:
    """Custom weighted binary cross-entropy loss emphasizing Gaussian center.
    
    Args:
        limit (float): Threshold for positive region.
        pos_weight (float): Weight for positive region.
    """
    def __init__(self, limit=0.5, pos_weight=10.0):
        self.limit = limit
        self.pos_weight = pos_weight

    def __call__(self, pred, target):
        """Compute weighted BCE loss.
        
        Args:
            pred (torch.Tensor): Predicted logits.
            target (torch.Tensor): Target heatmap.
        
        Returns:
            torch.Tensor: Loss value.
        """
        weights = torch.ones_like(target)
        # emphasize center of Gaussian
        weights[target > self.limit] = self.pos_weight
        bce = F.binary_cross_entropy_with_logits(
            pred, target, weight=weights, reduction='mean')
        return bce

class EfficientUNetLoss(WeightedBCEWithLogitsLoss):
    """EfficientUNet loss combining BCE, centroid, sparsity, and Dice losses.
    
    Args:
        limit (float): Threshold for positive region.
        pos_weight (float): Weight for positive region.
        centroid_weight (float): Weight for centroid distance loss.
        sparsity_weight (float): Weight for sparsity loss.
        dice_weight (float): Weight for Dice loss.
    """
    def __init__(self, limit=0.5, pos_weight=5.0, centroid_weight=2.5e-3, sparsity_weight=0.1, dice_weight=1.0):
        super().__init__(limit, pos_weight)
        self.centroid_weight = centroid_weight
        self.sparsity_weight = sparsity_weight
        self.dice_weight = dice_weight

    def __call__(self, pred, target, return_components=False):
        """Compute total loss for EfficientUNet.
        
        Args:
            pred (torch.Tensor): Predicted logits.
            target (torch.Tensor): Target heatmap.
            return_components (bool): If True, return (total_loss, components_dict).
        
        Returns:
            torch.Tensor or tuple: Loss value, or (loss, components_dict) if return_components=True.
        """
        bce_loss = super().__call__(pred, target)
        cdist_loss = self.centroid_distance_loss(pred, target)
        sparsity_loss_raw = torch.sigmoid(pred).mean()
        sparsity_loss = self.sparsity_weight * sparsity_loss_raw
        d_loss = self.dice_loss(pred, target)
        
        total_loss = bce_loss + (self.centroid_weight * cdist_loss) + sparsity_loss + (self.dice_weight * d_loss)
        
        if return_components:
            components = {
                'bce_loss': bce_loss.item(),
                'centroid_loss': cdist_loss.item(),
                'centroid_loss_weighted': (self.centroid_weight * cdist_loss).item(),
                'sparsity_loss': sparsity_loss_raw.item(),
                'sparsity_loss_weighted': sparsity_loss.item(),
                'dice_loss': d_loss.item(),
                'dice_loss_weighted': (self.dice_weight * d_loss).item(),
                'total_loss': total_loss.item()
            }
            return total_loss, components
        
        return total_loss

    @staticmethod
    def dice_loss(pred, target, smooth=1.0):
        """Compute Dice loss (1 - Dice coefficient).
        
        Args:
            pred (torch.Tensor): Logits from model.
            target (torch.Tensor): Ground truth mask.
            smooth (float): Smoothing factor.
        
        Returns:
            torch.Tensor: Dice loss value.
        """
        pred_probs = torch.sigmoid(pred)
        pred_flat = pred_probs.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1. - dice

    @staticmethod
    def centroid_distance_loss(pred, target):
        """Compute centroid distance loss between prediction and target.
        
        Args:
            pred (torch.Tensor): Predicted logits.
            target (torch.Tensor): Target heatmap.
        
        Returns:
            torch.Tensor: Mean centroid distance.
        """
        preds = get_centroids_per_sample(torch.sigmoid(pred))
        targets = get_centroids_per_sample(target)
        distances = []
        for p, t in zip(preds, targets):
            if p is not None and t is not None:
                x_p, y_p, _ = p
                x_t, y_t, _ = t
                dist = torch.sqrt((x_p - x_t) ** 2 + (y_p - y_t) ** 2 + 1e-8)
                distances.append(dist)
        if not distances:
            return torch.tensor(0.0).to(pred.device)
        return torch.stack(distances).mean()


class SiameseLoss(EfficientUNetLoss):
    """Siamese loss combining EfficientUNetLoss and RGB consistency loss.
    
    Args:
        limit (float): Threshold for positive region.
        pos_weight (float): Weight for positive region.
        centroid_weight (float): Weight for centroid distance loss.
        sparsity_weight (float): Weight for sparsity loss.
        dice_weight (float): Weight for Dice loss.
        rgb_weight (float): Weight for RGB consistency loss.
        rgb_sigma (float): Sigma for Gaussian in RGB loss.
        rgb_threshold (float): Threshold for mask in RGB loss.
    """
    def __init__(
        self,
        limit=0.5,
        pos_weight=10.0,
        centroid_weight=2.5e-3,
        sparsity_weight=1e-3,
        dice_weight=1.0,
        rgb_weight=5e-3,
        rgb_sigma=2.0,
        rgb_threshold=0.5
    ):
        super().__init__(limit, pos_weight, centroid_weight, sparsity_weight, dice_weight)
        self.rgb_weight = rgb_weight
        self.rgb_sigma = rgb_sigma
        self.rgb_threshold = rgb_threshold

    def __call__(self, output, target, template, search, return_components=False):
        """Compute total loss for Siamese model.
        
        Args:
            output (torch.Tensor): Predicted logits.
            target (torch.Tensor): Target heatmap.
            template (torch.Tensor): Template image.
            search (torch.Tensor): Search image.
            return_components (bool): If True, return (total_loss, components_dict).
        
        Returns:
            torch.Tensor or tuple: Loss value, or (loss, components_dict) if return_components=True.
        """
        if return_components:
            main_loss, main_components = super().__call__(output, target, return_components=True)
        else:
            main_loss = super().__call__(output, target, return_components=False)
        
        rgb_loss_raw = self.rgb_consistency_loss(template, search, output)
        rgb_loss = self.rgb_weight * rgb_loss_raw
        total_loss = main_loss + rgb_loss
        
        if return_components:
            components = main_components.copy()
            components['rgb_loss'] = rgb_loss_raw.item()
            components['rgb_loss_weighted'] = rgb_loss.item()
            components['total_loss'] = total_loss.item()
            return total_loss, components
        
        return total_loss

    def rgb_consistency_loss(self, template_img, search_img, pred_heatmap):
        """Compute RGB consistency loss between template and search images.
        
        Args:
            template_img (torch.Tensor): Template image tensor.
            search_img (torch.Tensor): Search image tensor.
            pred_heatmap (torch.Tensor): Predicted heatmap tensor.
        
        Returns:
            torch.Tensor: RGB consistency loss value.
        """
        B, _, H, W = template_img.shape
        device = template_img.device
        # Create fixed centered Gaussian for all batch
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H - 1, H, device=device),
            torch.linspace(0, W - 1, W, device=device),
            indexing='ij'
        )
        center_y = (H - 1) / 2
        center_x = (W - 1) / 2
        gaussian = torch.exp(-((grid_x - center_x)**2 + (grid_y - center_y)**2) / (2 * self.rgb_sigma**2))
        gaussian /= gaussian.sum() + 1e-8
        gaussian = gaussian[None, None, :, :]  # shape (1, 1, H, W)
        loss = 0.0
        for i in range(B):
            # Mask and normalize predicted heatmap
            mask = (pred_heatmap[i] > self.rgb_threshold).float()
            weighted_mask = pred_heatmap[i] * mask
            weighted_mask /= weighted_mask.sum() + 1e-8  # (1, H, W)
            # Compute mean RGB in search
            rgb_search = (search_img[i] * weighted_mask).view(3, -1).sum(dim=1)
            # Compute mean RGB in template using Gaussian
            rgb_template = (template_img[i] * gaussian[0]).view(3, -1).sum(dim=1)
            loss += F.mse_loss(rgb_search, rgb_template)
        return loss / B


def _sort_samples(val_results, num_samples):
    sorted_results = sorted(
        val_results, key=lambda r: r['confidence'], reverse=True)
    worst = sorted_results[-num_samples:][::-1]
    best = sorted_results[:num_samples]
    mid_start = len(sorted_results) // 2 - num_samples // 2
    mid_end = mid_start + num_samples
    middle = sorted_results[mid_start:mid_end]
    return worst + middle + best


class BaseVisualizer:
    """Base class for visualizers used in training visualization."""
    def __init__(self, model, device, val_results, stage, epoch, output_dir="vis_epochs"):
        """Initialize the visualizer.
        
        Args:
            model: Model instance.
            device: Torch device.
            val_results (list): Validation results.
            stage (int): Training stage.
            epoch (int): Epoch number.
            output_dir (str): Output directory for visualizations.
        """
        self.model = model
        self.device = device
        self.val_results = val_results
        self.stage = stage
        self.epoch = epoch
        self.output_dir = output_dir

    def _save_fig(self, fig, subdir, prefix):
        """Save a matplotlib figure to disk.
        
        Args:
            fig: Matplotlib figure.
            subdir (str): Subdirectory for saving.
            prefix (str): Filename prefix.
        """
        os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        path = os.path.join(
            self.output_dir,
            subdir,
            f"{prefix}_stage_{self.stage:03d}_epoch_{self.epoch:03d}.png"
        )
        fig.savefig(path, dpi=200)
        plt.close(fig)


class SiameseTrackingVisualizer(BaseVisualizer):
    """Visualizer for Siamese tracking model performance and activations."""
    def performance(self, num_samples=5, save=True):
        """Visualize performance samples for Siamese tracking.
        
        Args:
            num_samples (int): Number of samples per group (worst, middle, best).
        """
        samples = _sort_samples(self.val_results, num_samples)
        fig, axes = plt.subplots(
            len(samples), 3, figsize=(9, 3 * len(samples)))


        for i, result in enumerate(samples):
            template_img = TF.to_pil_image(denormalize(result['template']))
            search_img = TF.to_pil_image(denormalize(result['search']))
            search_np = TF.to_tensor(search_img).permute(1, 2, 0).numpy()

            pred, gt = result['pred_heatmap'], result['gt_heatmap']
            pred = pred.detach().cpu()
            gt = gt.detach().cpu()
            pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
            diff_norm = np.abs(pred_norm.numpy() - gt_norm.numpy())

            overlay = np.clip(0.6 * search_np + 0.4 *
                              plt.cm.jet(pred_norm)[..., :3], 0, 1)
            diff_rgb = plt.cm.magma(diff_norm)[..., :3]

            xg, yg = result['gt_centroid']
            xp, yp = result['pred_centroid']
            confidence = result['confidence']

            axes[i, 0].imshow(template_img)
            axes[i, 0].set_title(f"Template idx {i}")

            axes[i, 1].imshow(overlay)
            axes[i, 1].scatter([xp], [yp], c='red', marker='x', label='Pred')
            axes[i, 1].scatter([xg], [yg], c='green', marker='o', label='GT')
            axes[i, 1].set_title(f"Search | Conf: {confidence:.2f}")
            axes[i, 1].legend()

            axes[i, 2].imshow(diff_rgb)
            axes[i, 2].set_title("Abs Diff")

            for ax in axes[i]:
                ax.axis("off")

        plt.tight_layout()
        if save:
            self._save_fig(plt.gcf(), "performance", "epoch")
        
        return fig

    def activations(self, num_samples=3, save=True):
        """Visualize activations for Siamese tracking model.
        
        Args:
            num_samples (int): Number of samples to visualize.
        """
        output_dir = os.path.join(self.output_dir, "activations")
        os.makedirs(output_dir, exist_ok=True)

        stages = ['enc3', 'enc4', 'enc5', 'up4',
                  'up3', 'up2', 'up1', 'up0', 'out']
        channels_per_stage = 3

        activations = {}
        for name in stages:
            layer = getattr(self.model, name)
            layer.register_forward_hook(
                lambda m, i, o, n=name: activations.update({n: o.detach().cpu()}))

        samples = _sort_samples(self.val_results, num_samples)
        n_cols = 1 + channels_per_stage * len(stages)
        fig, axs = plt.subplots(len(samples), n_cols, figsize=(
            n_cols * 2.5, len(samples) * 3))
        axs = axs if len(samples) > 1 else axs[None, :]

        self.model.eval()
        for row, sample in enumerate(samples):
            template = sample['template'].unsqueeze(0).to(self.device)
            search = sample['search'].unsqueeze(0).to(self.device)
            heatmap = sample['pred_heatmap'].numpy()

            with torch.no_grad():
                _ = self.model(template, search)

            overlay = denormalize(search[0]).permute(1, 2, 0).cpu().numpy()
            overlay[..., 0] = np.clip(overlay[..., 0] + 0.5 * heatmap, 0, 1)

            axs[row, 0].imshow(overlay)
            axs[row, 0].scatter([sample['gt_centroid'][0]], [
                                sample['gt_centroid'][1]], c='green', marker='o')
            axs[row, 0].scatter([sample['pred_centroid'][0]], [
                                sample['pred_centroid'][1]], c='red', marker='x')
            axs[row, 0].set_title(f"Conf: {sample['confidence']:.2f}")
            axs[row, 0].axis('off')

            col = 1
            for stage in stages:
                act = activations[stage][0]
                for ch in range(channels_per_stage):
                    if ch < act.shape[0]:
                        axs[row, col].imshow(act[ch], cmap='viridis')
                        axs[row, col].set_title(f'{stage} | Ch {ch}')
                    axs[row, col].axis('off')
                    col += 1

        plt.tight_layout()
        if save:
            self._save_fig(plt.gcf(), "activations", "activation")
        
        return fig


class EfficientUNetVisualizer(BaseVisualizer):
    """Visualizer for EfficientUNet model performance and activations."""
    def performance(self, num_samples=5, save=True):
        """Visualize performance samples for EfficientUNet.
        
        Args:
            num_samples (int): Number of samples per group (worst, middle, best).
        """
        samples = _sort_samples(self.val_results, num_samples)
        fig, axes = plt.subplots(
            len(samples), 3, figsize=(9, 3 * len(samples)))


        for i, result in enumerate(samples):
            search_img = TF.to_pil_image(denormalize(result['search']))
            search_np = TF.to_tensor(search_img).permute(1, 2, 0).numpy()

            pred, gt = result['pred_heatmap'], result['gt_heatmap']
            pred = pred.detach().cpu()
            gt = gt.detach().cpu()
            pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
            diff_norm = np.abs(pred_norm.numpy() - gt_norm.numpy())

            overlay = np.clip(0.6 * search_np + 0.4 *
                              plt.cm.jet(pred_norm)[..., :3], 0, 1)
            diff_rgb = plt.cm.magma(diff_norm)[..., :3]

            xg, yg = result['gt_centroid']
            xp, yp = result['pred_centroid']
            confidence = result['confidence']

            axes[i, 0].imshow(overlay)
            axes[i, 0].scatter([xp], [yp], c='red', marker='x', label='Pred')
            axes[i, 0].scatter([xg], [yg], c='green', marker='o', label='GT')
            axes[i, 0].set_title(f"Search | Conf: {confidence:.2f}")
            axes[i, 0].legend()

            axes[i, 1].imshow(diff_rgb)
            axes[i, 1].set_title("Abs Diff")

            for ax in axes[i]:
                ax.axis("off")

        plt.tight_layout()

        if save:
            self._save_fig(plt.gcf(), "performance", "stage")
        
        return fig

    def activations(self, num_samples=3, save=True):
        """Visualize activations for EfficientUNet model.
        
        Args:
            num_samples (int): Number of samples to visualize.
        """
        output_dir = os.path.join(self.output_dir, "activations")
        os.makedirs(output_dir, exist_ok=True)

        stages = ['enc1', 'enc2', 'enc3', 'enc4', 'enc5',
                  'up4', 'up3', 'up2', 'up1', 'up0', 'out']
        channels_per_stage = 3

        activations = {}
        for name in stages:
            layer = getattr(self.model, name)
            layer.register_forward_hook(
                lambda m, i, o, n=name: activations.update({n: o.detach().cpu()}))

        samples = _sort_samples(self.val_results, num_samples)
        n_cols = 1 + channels_per_stage * len(stages)
        fig, axs = plt.subplots(len(samples), n_cols, figsize=(
            n_cols * 2.5, len(samples) * 3))
        axs = axs if len(samples) > 1 else axs[None, :]

        self.model.eval()
        for row, sample in enumerate(samples):
            search = sample['search'].unsqueeze(0).to(self.device)
            heatmap = sample['pred_heatmap'].numpy()

            with torch.no_grad():
                _ = self.model(search)

            overlay = denormalize(search[0]).permute(1, 2, 0).cpu().numpy()
            overlay[..., 0] = np.clip(overlay[..., 0] + 0.5 * heatmap, 0, 1)

            axs[row, 0].imshow(overlay)
            axs[row, 0].scatter([sample['gt_centroid'][0]], [
                                sample['gt_centroid'][1]], c='green', marker='o')
            axs[row, 0].scatter([sample['pred_centroid'][0]], [
                                sample['pred_centroid'][1]], c='red', marker='x')
            axs[row, 0].set_title(f"Conf: {sample['confidence']:.2f}")
            axs[row, 0].axis('off')
            axs[row, 0].legend()

            col = 1
            for stage in stages:
                act = activations[stage][0]
                for ch in range(channels_per_stage):
                    if ch < act.shape[0]:
                        axs[row, col].imshow(act[ch], cmap='viridis')
                        axs[row, col].set_title(f'{stage} | Ch {ch}')
                    axs[row, col].axis('off')
                    col += 1

        plt.tight_layout()
        if save:
            self._save_fig(plt.gcf(), "activations", "activation")
        
        return fig
