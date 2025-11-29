import argparse

import torch

import torch.nn as nn

# Import training utilities
from aegear.nn.training import (
    train,
    setup_logging,
    get_device,
    setup_model,
    get_default_training_stages,
    load_training_stages,
    EfficientUNetLoss,
    SiameseLoss
)

from aegear.nn.datasets import load_dataset_from_shards

# Helper to parse scheduler params
def parse_scheduler_params(param_str):
    params = {}
    if param_str:
        for kv in param_str.split(","):
            if "=" in kv:
                k, v = kv.split("=")
                try:
                    params[k.strip()] = float(v)
                except ValueError:
                    params[k.strip()] = v.strip()
    return params

def parse_loss_params(param_str):
    params = {}
    if param_str:
        for kv in param_str.split(","):
            k, v = kv.split("=")
            try:
                params[k.strip()] = float(v)
            except ValueError:
                params[k.strip()] = v.strip()
    return params

def main():

    parser = argparse.ArgumentParser(description="Train EfficientUNet or SiameseTracker with CLI configuration.")

    parser.add_argument("--model-type", choices=["efficient_unet", "siamese"], required=True)
    parser.add_argument("--data-manifest", type=str, required=True, help="Path to dataset manifest JSON file.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-ratio", type=float, default=0.85, help="Ratio of training data when splitting dataset.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader worker threads.")
    parser.add_argument("--gaussian-sigma", type=float, default=15.0)
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V1")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--use-best-model", action="store_true")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--pretrained-model-dir", type=str, default="models/")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training. Applied if no training stages are specified.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training. Applied if no training stages are specified.")
    parser.add_argument("--epoch-vis", type=str, default="vis_epochs")
    parser.add_argument("--epoch-save-interval", type=int, default=1)
    parser.add_argument("--training-stages", type=str, help="Path to training stages JSON config.")
    parser.add_argument("--loss-params", type=str, help="Comma-separated key=value pairs for loss function.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps", "auto"], default="auto")
    parser.add_argument("--config", type=str, help="Path to config file for HPO integration.")
    parser.add_argument("--weight-decay", type=float, default=5e-3, help="Weight decay for optimizer.")
    parser.add_argument("--cbam", action="store_true", help="Use CBAM in model.")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function to use (relu, leakyrelu, gelu, etc.)")
    parser.add_argument("--clearml-task", type=str, default="", help="Name for ClearML task. If empty, ClearML is disabled.")
    parser.add_argument("--clearml-project", type=str, default="aegear", help="Name for ClearML project. Default is 'aegear'.")
    parser.add_argument("--use-visualizer", action="store_true", help="Enable visualizer plots and logging.")
    parser.add_argument("--scheduler-type", type=str, choices=["ReduceLROnPlateau", "OneCycleLR", "StepLR", "CosineAnnealingLR"], default=None, help="LR scheduler type.")
    parser.add_argument("--scheduler-params", type=str, default=None, help="Comma-separated key=value pairs for scheduler params.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--autodownload", action="store_true", help="Automatically download dataset shards if not present.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output during dataset loading.")

    args = parser.parse_args()

    # ClearML integration (optional)
    task = None
    if args.clearml_task:
        try:
            from clearml import Task
            task = Task.init(project_name=args.clearml_project, task_name=args.clearml_task)
            task.connect(args)
        except ImportError:
            print("ClearML not installed, skipping experiment tracking.")

    logger = setup_logging()
    logger.info(f"Starting training for {args.model_type}")

    # Device selection
    print("\n" + "="*60)
    print("DEVICE SELECTION DEBUG")
    print("="*60)
    print(f"args.device from CLI: {args.device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if args.device == "auto":
        device = get_device()
        print(f"Device selection mode: AUTO")
    else:
        device = args.device if args.device != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device selection mode: EXPLICIT ({args.device})")
    
    print(f"Selected device: {device}")
    print("="*60 + "\n")
    
    # Validate device - fail early if CUDA was expected but unavailable
    if args.device in ["cuda", "auto"] and device == "cpu" and not torch.cuda.is_available():
        print("\n" + "="*60)
        print("DEVICE VALIDATION FAILED")
        print("="*60)
        print("CUDA device was requested but is unavailable.")
        print("This indicates a machine-level issue (likely driver problems).")
        print("\nExiting with code 42 to signal retry on a different machine.")
        print("="*60 + "\n")
        import sys
        sys.exit(42)

    # Dataset loading
    train_loader, val_loader = load_dataset_from_shards(
        args.data_manifest,
        gaussian_sigma=args.gaussian_sigma,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        autodownload=args.autodownload,
        verbose=args.verbose
    )

    try:
        train_len = len(train_loader.dataset)
    except TypeError:
        train_len = "unknown"
    try:
        val_len = len(val_loader.dataset)
    except TypeError:
        val_len = "unknown"

    try:
        train_batches = len(train_loader)
    except TypeError:
        train_batches = "unknown"
    try:
        val_batches = len(val_loader)
    except TypeError:
        val_batches = "unknown"

    print("Dataset diagnostics:")
    print(f"  Train samples: {train_len}")
    print(f"  Train batches/epoch: {train_batches}")
    print(f"  Val samples: {val_len}")
    print(f"  Val batches/epoch: {val_batches}")

    # Model setup
    # Map activation string to torch.nn activation class
    activation_map = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "prelu": nn.PReLU,
    }
    activation_cls = activation_map.get(args.activation.lower(), nn.ReLU)

    # Setup EfficientUNet backbone
    model_backbone = setup_model(
        weights=args.weights,
        continue_training=args.continue_training,
        use_best_model=args.use_best_model,
        model_dir=args.model_dir,
        pretrained_model_dir=args.pretrained_model_dir,
        device=device,
        use_cbam=args.cbam,
        activation=activation_cls
    )

    if args.model_type == "efficient_unet":
        model = model_backbone
    elif args.model_type == "siamese":
        from aegear.nn.model import SiameseTracker
        model = SiameseTracker(model_backbone)
        model.to(device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Training stages
    if args.training_stages:
        training_stages = load_training_stages(model, args.training_stages)
    else:
        training_stages = get_default_training_stages(args.model_type, args.epochs, args.lr)

    # Loss function
    loss_params = parse_loss_params(args.loss_params)
    if args.model_type == "efficient_unet":
        loss_fn = EfficientUNetLoss(**loss_params)
    else:
        loss_fn = SiameseLoss(**loss_params)

    # Scheduler config
    scheduler_config = None
    if args.scheduler_type:
        scheduler_config = {
            'type': args.scheduler_type,
            'params': parse_scheduler_params(args.scheduler_params)
        }

    if scheduler_config and scheduler_config.get('type') == 'OneCycleLR':
        # These are automatically set in the training loop for OneCycleLR, so just in case remove them
        scheduler_config['params'].pop('epochs', None)
        scheduler_config['params'].pop('steps_per_epoch', None)

    train(
        model,
        train_loader,
        val_loader,
        device,
        args.model_dir,
        args.checkpoint_dir,
        args.epoch_vis,
        training_stages,
        loss_fn=loss_fn,
        epoch_save_interval=args.epoch_save_interval,
        model_type=args.model_type,
        use_visualizer=args.use_visualizer,
        weight_decay=args.weight_decay,
        clearml_task=task,
        scheduler_config=scheduler_config
    )

if __name__ == "__main__":
    """
    Example usage:
    python tools/train.py `
        --model-type efficient_unet `
        --batch-size 128 `
        --gaussian-sigma 15.0 `
        --num-workers 4 `
        --verbose `
        --use-visualizer `
        --data-manifest data/training/tracking_merged/manifest.json `
        --model-dir data/training/models/efficient_unet `
        --checkpoint-dir data/training/models/efficient_unet/checkpoints `
        --epoch-vis data/training/models/efficient_unet/epoch_vis `
        --epoch-save-interval 3 `
        --scheduler-type ReduceLROnPlateau `
        --scheduler-params mode=min,factor=0.5,patience=3
    """
    main()

