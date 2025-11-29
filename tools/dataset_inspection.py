#!/usr/bin/env python3
"""
Dataset Inspection Tool for Model Evaluation

This script provides a CLI for inspecting and evaluating detection and tracking
models using FiftyOne visualization.
"""

import argparse
import os
import sys
import torch

from aegear.nn.model import SiameseTracker, EfficientUNet
from aegear.nn.datasets import CachedTrackingDataset, CachedDetectionDataset
from aegear.utils import get_latest_model_path, download_dataset, load_model_with_weights
from aegear.visualization import (
    TrackingDatasetBuilder,
    DetectionDatasetBuilder,
    launch_fiftyone_app
)


# Default configuration
DEFAULT_DATASET_DIR = "data/training/cache"
DEFAULT_MODELS_DIR = "models/"
DEFAULT_GAUSSIAN_SIGMA = 12.0
DEFAULT_BATCH_SIZE = 128


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect and evaluate detection/tracking models with FiftyOne",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate tracking model on default dataset
  python dataset_inspection.py tracking --dataset-name 4_per_23
  
  # Evaluate detection model with custom batch size
  python dataset_inspection.py detection --dataset-name my_detection --batch-size 64
  
  # Use custom dataset path
  python dataset_inspection.py tracking --custom-path /path/to/dataset
  
  # Specify model path explicitly
  python dataset_inspection.py tracking --dataset-name 4_per_23 --model-path models/model_siamese_2025-01-15.pth
        """
    )
    
    # Required arguments
    parser.add_argument(
        "mode",
        type=str,
        choices=["tracking", "detection"],
        help="Type of model/dataset to evaluate"
    )
    
    # Dataset arguments
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset-name",
        type=str,
        help="Name of the dataset (expects dataset in default cache structure)"
    )
    dataset_group.add_argument(
        "--custom-path",
        type=str,
        help="Custom path to dataset (must follow CachedDataset format)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model checkpoint. If not provided, uses latest model from models directory"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory containing model checkpoints (default: {DEFAULT_MODELS_DIR})"
    )
    
    # Dataset settings
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=DEFAULT_DATASET_DIR,
        help=f"Base directory for datasets (default: {DEFAULT_DATASET_DIR})"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip automatic dataset download from GCS"
    )
    
    # Training/inference parameters
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=DEFAULT_GAUSSIAN_SIGMA,
        help=f"Gaussian sigma for heatmap generation (default: {DEFAULT_GAUSSIAN_SIGMA})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for inference (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)"
    )
    
    # FiftyOne settings
    parser.add_argument(
        "--fiftyone-name",
        type=str,
        help="Name for the FiftyOne dataset (default: auto-generated)"
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Don't launch FiftyOne app automatically"
    )
    
    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to run inference on (default: auto)"
    )
    
    return parser.parse_args()


def get_device(device_arg):
    """Determine the device to use for inference."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_dataset(args):
    """Load the appropriate dataset based on arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
        
    Returns
    -------
    Dataset
        Loaded dataset object
    """
    # Determine dataset path
    if args.custom_path:
        dataset_path = args.custom_path
    else:
        # Download dataset if needed
        if not args.skip_download:
            download_dataset(args.dataset_dir, args.mode)
        
        dataset_path = os.path.join(
            args.dataset_dir,
            args.mode,
            args.dataset_name,
            "val"
        )
    
    # Validate dataset path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    # Load appropriate dataset
    if args.mode == "tracking":
        dataset = CachedTrackingDataset(
            dataset_path,
            gaussian_sigma=args.gaussian_sigma
        )
    else:  # detection
        dataset = CachedDetectionDataset(
            dataset_path,
            gaussian_sigma=args.gaussian_sigma
        )
    
    print(f"Loaded {args.mode} dataset from: {dataset_path}")
    print(f"Dataset size: {len(dataset)} samples")
    
    return dataset


def load_model(args, device):
    """Load the appropriate model based on arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    device : str
        Device to load model on
        
    Returns
    -------
    torch.nn.Module
        Loaded model
    """
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        # Get latest model from models directory
        model_prefix = "model_siamese" if args.mode == "tracking" else "model_efficient_unet"
        model_path = get_latest_model_path(args.models_dir, model_prefix)
        
        if model_path is None:
            raise FileNotFoundError(
                f"No model found matching '{model_prefix}' in {args.models_dir}"
            )
    
    # Validate model path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    # Load appropriate model
    if args.mode == "tracking":
        model_class = SiameseTracker
    else:  # detection
        model_class = EfficientUNet
    
    model = load_model_with_weights(model_class, model_path, device)
    print(f"Loaded model from: {model_path}")
    
    return model


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Setup
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load dataset and model
    try:
        dataset = load_dataset(args)
        model = load_model(args, device)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during loading: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create FiftyOne dataset name
    if args.fiftyone_name:
        fo_name = args.fiftyone_name
    else:
        dataset_identifier = args.dataset_name or os.path.basename(args.custom_path)
        fo_name = f"{args.mode}-eval-{dataset_identifier}"
    
    # Build FiftyOne dataset
    try:
        if args.mode == "tracking":
            builder = TrackingDatasetBuilder(
                dataset=dataset,
                model=model,
                device=device,
                img_size=dataset.output_size
            )
        else:  # detection
            builder = DetectionDatasetBuilder(
                dataset=dataset,
                model=model,
                device=device,
                img_size=dataset.output_size
            )
        
        fo_dataset = builder.build_dataset(
            fo_dataset_name=fo_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        print(f"\nFiftyOne dataset '{fo_name}' created successfully!")
        print(f"Total samples: {len(fo_dataset)}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Launch FiftyOne app
    if not args.no_launch:
        try:
            session = launch_fiftyone_app(fo_dataset)
            session.wait()
        except KeyboardInterrupt:
            print("\nClosing FiftyOne app...")
    else:
        print(f"\nTo view the dataset, run:")
        print(f"  fiftyone app launch {fo_name}")


if __name__ == "__main__":
    main()