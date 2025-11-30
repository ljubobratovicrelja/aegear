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
    load_training_stages_from_config,
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


def parse_training_stages_arg(stages_arg):
    """
    Parse training stages from command-line argument.
    
    Supports multiple formats:
    1. Path to JSON file: "/path/to/stages.json"
    2. Path to YAML file: "/path/to/stages.yaml"
    3. JSON string: '{"stages": [{"freeze_layers": ["enc1"], "epochs": 10, "lr": 0.001}]}'
    4. Inline format: "stage1:freeze=enc1,enc2:epochs=10:lr=0.001;stage2:freeze=enc1:epochs=5:lr=0.0001"
    
    Returns:
        Either a path string (for JSON) or a list of stage dicts (for inline format, YAML, or JSON string).
    """
    import os
    import json
    
    if not stages_arg:
        return None
    
    # Try to parse as JSON string first (for config passed from YAML)
    if stages_arg.strip().startswith(('{', '[')):
        try:
            parsed = json.loads(stages_arg)
            # Handle both direct list and dict with 'stages' key
            if isinstance(parsed, dict) and 'stages' in parsed:
                return parsed['stages']
            elif isinstance(parsed, list):
                return parsed
            else:
                # Invalid structure, fall through
                pass
        except json.JSONDecodeError:
            # Not valid JSON, try other formats
            pass
    
    # Check if it's a file path
    if os.path.isfile(stages_arg):
        # Return path for JSON files (handled by load_training_stages)
        if stages_arg.endswith('.json'):
            return stages_arg
        # Parse YAML files directly
        elif stages_arg.endswith('.yaml') or stages_arg.endswith('.yml'):
            import yaml
            with open(stages_arg, 'r') as f:
                stages_data = yaml.safe_load(f)
            # Handle both direct list and dict with 'stages' key
            if isinstance(stages_data, dict) and 'stages' in stages_data:
                return stages_data['stages']
            elif isinstance(stages_data, list):
                return stages_data
            else:
                raise ValueError(f"YAML stages file must contain a list or dict with 'stages' key")
        else:
            # Assume JSON if no extension or unknown extension
            return stages_arg
    
    # Parse inline format: \"stage1:freeze=enc1,enc2:epochs=10:lr=0.001;stage2:...\"\n    stages = []
    for stage_str in stages_arg.split(';'):
        stage_dict = {}
        for part in stage_str.split(':'):
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'freeze':
                    # freeze_layers is a list
                    stage_dict['freeze_layers'] = [v.strip() for v in value.split(',')]
                elif key == 'epochs':
                    stage_dict['epochs'] = int(value)
                elif key == 'lr':
                    stage_dict['lr'] = float(value)
                elif key == 'name':
                    stage_dict['name'] = value
                else:
                    # Store other parameters as-is
                    try:
                        stage_dict[key] = float(value)
                    except ValueError:
                        stage_dict[key] = value
        
        if stage_dict:  # Only add non-empty stages
            stages.append(stage_dict)
    
    return stages if stages else None


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

    # Print comprehensive training configuration summary
    print("\n" + "="*80)
    print("AEGEAR TRAINING CONFIGURATION")
    print("="*80)
    
    # Task Information
    print("\n[TASK INFORMATION]")
    print("-" * 80)
    if args.clearml_task:
        print(f"  ClearML Project:      {args.clearml_project}")
        print(f"  ClearML Task:         {args.clearml_task}")
    else:
        print(f"  ClearML:              Disabled")
    print(f"  Model Type:           {args.model_type}")
    print(f"  Data Manifest:        {args.data_manifest}")
    print(f"  Model Directory:      {args.model_dir}")
    print(f"  Checkpoint Directory: {args.checkpoint_dir}")
    
    # Model Architecture
    print("\n[MODEL ARCHITECTURE]")
    print("-" * 80)
    print(f"  Activation Function:  {args.activation}")
    print(f"  CBAM Attention:       {'Enabled' if args.cbam else 'Disabled'}")
    print(f"  Pretrained Weights:   {args.weights}")
    print(f"  Continue Training:    {'Yes' if args.continue_training else 'No'}")
    print(f"  Use Best Model:       {'Yes' if args.use_best_model else 'No'}")
    
    # Device selection
    print("\n[DEVICE SELECTION]")
    print("-" * 80)
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
    print("-" * 80)
    
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

    # Dataset Configuration
    print("\n[DATASET CONFIGURATION]")
    print("-" * 80)
    print(f"  Batch Size:           {args.batch_size}")
    print(f"  Train Ratio:          {args.train_ratio:.2%}")
    print(f"  Num Workers:          {args.num_workers}")
    print(f"  Gaussian Sigma:       {args.gaussian_sigma}")
    print(f"  Random Seed:          {args.seed}")
    print(f"  Auto-download:        {'Enabled' if args.autodownload else 'Disabled'}")
    print(f"  Verbose Mode:         {'Enabled' if args.verbose else 'Disabled'}")
    print(f"\n  Dataset Statistics:")
    print(f"    Train samples:      {train_len}")
    print(f"    Train batches:      {train_batches}")
    print(f"    Val samples:        {val_len}")
    print(f"    Val batches:        {val_batches}")

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
        parsed_stages = parse_training_stages_arg(args.training_stages)
        
        if isinstance(parsed_stages, str):
            # It's a JSON file path
            training_stages = load_training_stages(model, parsed_stages)
        elif isinstance(parsed_stages, list):
            # It's already parsed (from YAML or inline format)
            training_stages = load_training_stages_from_config(model, parsed_stages)
        else:
            raise ValueError(f"Unexpected training stages format: {type(parsed_stages)}")
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

    # Print Training Stages Configuration
    print("\n[TRAINING STAGES]")
    print("-" * 80)
    total_epochs = sum(stage['epochs'] for stage in training_stages)
    print(f"  Total Stages:         {len(training_stages)}")
    print(f"  Total Epochs:         {total_epochs}")
    print()
    
    for idx, stage in enumerate(training_stages, 1):
        stage_name = stage.get('name', f'Stage {idx}')
        print(f"  Stage {idx}: {stage_name}")
        print(f"    Epochs:             {stage['epochs']}")
        print(f"    Learning Rate:      {stage['lr']:.6f}")
        
        # Show frozen layers
        if 'freeze_layers' in stage and stage['freeze_layers']:
            # Get layer names (they might be objects at this point)
            frozen_layer_names = []
            for layer in stage['freeze_layers']:
                if hasattr(layer, '__class__'):
                    # It's an actual layer object, try to get its name from the model
                    layer_name = 'unknown'
                    for name, module in model.named_modules():
                        if module is layer:
                            layer_name = name
                            break
                    frozen_layer_names.append(layer_name)
                else:
                    # It's a string
                    frozen_layer_names.append(str(layer))
            
            print(f"    Frozen Layers:      {', '.join(frozen_layer_names) if frozen_layer_names else 'None'}")
        else:
            print(f"    Frozen Layers:      None (full model training)")
        print()
    
    # Optimizer Configuration
    print("\n[OPTIMIZER & SCHEDULER]")
    print("-" * 80)
    print(f"  Optimizer:            Adam")
    print(f"  Weight Decay:         {args.weight_decay:.6f}")
    if scheduler_config:
        print(f"  LR Scheduler:         {scheduler_config['type']}")
        if scheduler_config['params']:
            print(f"  Scheduler Params:")
            for key, value in scheduler_config['params'].items():
                print(f"    {key:20s}: {value}")
    else:
        print(f"  LR Scheduler:         ReduceLROnPlateau (default)")
        print(f"  Scheduler Params:")
        print(f"    mode                : min")
        print(f"    factor              : 0.5")
        print(f"    patience            : 3")
    
    # Loss Function Configuration
    print("\n[LOSS FUNCTION]")
    print("-" * 80)
    loss_class_name = loss_fn.__class__.__name__
    print(f"  Loss Type:            {loss_class_name}")
    if loss_params:
        print(f"  Loss Parameters:")
        for key, value in loss_params.items():
            if isinstance(value, float):
                print(f"    {key:20s}: {value:.4f}")
            else:
                print(f"    {key:20s}: {value}")
    else:
        print(f"  Loss Parameters:      (using defaults)")
    
    # Training Options
    print("\n[TRAINING OPTIONS]")
    print("-" * 80)
    print(f"  Visualizer:           {'Enabled' if args.use_visualizer else 'Disabled'}")
    print(f"  Epoch Vis Dir:        {args.epoch_vis}")
    print(f"  Checkpoint Interval:  Every {args.epoch_save_interval} epoch(s)")
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

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
    # Example usage:
    # python tools/train.py \
    #     --model-type efficient_unet \
    #     --batch-size 128 \
    #     --gaussian-sigma 15.0 \
    #     --num-workers 4 \
    #     --verbose \
    #     --use-visualizer \
    #     --data-manifest data/training/tracking_merged/manifest.json \
    #     --model-dir data/training/models/efficient_unet \
    #     --checkpoint-dir data/training/models/efficient_unet/checkpoints \
    #     --epoch-vis data/training/models/efficient_unet/epoch_vis \
    #     --epoch-save-interval 3 \
    #     --scheduler-type ReduceLROnPlateau \
    #     --scheduler-params mode=min,factor=0.5,patience=3
    main()

