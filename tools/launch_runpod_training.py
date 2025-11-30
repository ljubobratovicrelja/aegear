#!/usr/bin/env python3
"""
RunPod Training Launcher

Launch a RunPod pod to run aegear training with automatic termination on completion.
Supports Docker Hub authentication to avoid rate limits.

Usage:
    python launch_runpod_training.py --model-type efficient_unet --data-manifest /workspace/data/manifest.json

Environment variables:
    RUNPOD_API_TOKEN: RunPod API token (required)
    DOCKERHUB_USERNAME: Docker Hub username for authentication (optional)
    DOCKERHUB_PAT: Docker Hub personal access token (optional)
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
from aegear.nn.ops.runpod_launcher import RunPodLauncher
from aegear.nn.ops import build_training_env_vars


def expand_task_name(task_name):
    """Expand wildcards in task name with datetime values.
    
    Supported wildcards:
        {date} or {DATE} -> YYYY-MM-DD (e.g., 2025-11-30)
        {time} or {TIME} -> HH-MM-SS (e.g., 14-30-45)
        {datetime} or {DATETIME} -> YYYYMMDD_HHMMSS (e.g., 20251130_143045)
        {timestamp} or {TIMESTAMP} -> Unix timestamp (e.g., 1732976400)
    
    Args:
        task_name: Task name string with optional wildcards
        
    Returns:
        str: Expanded task name with wildcards replaced
    
    Examples:
        >>> expand_task_name("training_{date}")
        "training_2025-11-30"
        >>> expand_task_name("exp_{datetime}_v1")
        "exp_20251130_143045_v1"
    """
    if not task_name:
        return task_name
    
    now = datetime.now()
    
    # Replace all wildcard variants (case-insensitive)
    replacements = {
        '{date}': now.strftime('%Y-%m-%d'),
        '{DATE}': now.strftime('%Y-%m-%d'),
        '{time}': now.strftime('%H-%M-%S'),
        '{TIME}': now.strftime('%H-%M-%S'),
        '{datetime}': now.strftime('%Y%m%d_%H%M%S'),
        '{DATETIME}': now.strftime('%Y%m%d_%H%M%S'),
        '{timestamp}': str(int(now.timestamp())),
        '{TIMESTAMP}': str(int(now.timestamp())),
    }
    
    expanded_name = task_name
    for wildcard, value in replacements.items():
        expanded_name = expanded_name.replace(wildcard, value)
    
    return expanded_name


def load_config(config_path):
    """Load training configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch RunPod training job for Aegear",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using configuration file
  python launch_runpod_training.py --config config/training_config.yaml
  
  # Using configuration file with overrides
  python launch_runpod_training.py --config config/training_config.yaml --epochs 30 --lr 0.01
  
  # Basic usage without config file
  python launch_runpod_training.py \\
      --task-name efficient_unet_exp1 \\
      --model-type efficient_unet \\
      --data-manifest /workspace/data/manifest.json \\
      --model-dir /workspace/models/efficient_unet \\
      --checkpoint-dir /workspace/models/efficient_unet/checkpoints
  
  # With ClearML tracking
  python launch_runpod_training.py \\
      --task-name siamese_hpo_001 \\
      --model-type siamese \\
      --data-manifest /workspace/data/manifest.json \\
      --model-dir /workspace/models/siamese \\
      --checkpoint-dir /workspace/models/siamese/checkpoints \\
      --clearml-task siamese_hpo_001 \\
      --clearml-project aegear-hpo

Environment Variables:
  RUNPOD_API_TOKEN         RunPod API token (required)
  DOCKERHUB_USERNAME       Docker Hub username for auth (optional)
  DOCKERHUB_PAT           Docker Hub PAT for auth (optional)
  CLEARML_API_ACCESS_KEY  ClearML API access key (optional)
  CLEARML_API_SECRET_KEY  ClearML API secret key (optional)
        """
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file (CLI args override config file values)"
    )
    
    # RunPod configuration
    runpod_group = parser.add_argument_group("RunPod Configuration")
    runpod_group.add_argument(
        "--api-token",
        type=str,
        default=os.getenv("RUNPOD_API_TOKEN"),
        help="RunPod API token (default: $RUNPOD_API_TOKEN)"
    )
    runpod_group.add_argument(
        "--docker-username",
        type=str,
        default=os.getenv("DOCKERHUB_USERNAME"),
        help="Docker Hub username (default: $DOCKERHUB_USERNAME)"
    )
    runpod_group.add_argument(
        "--docker-pat",
        type=str,
        default=os.getenv("DOCKERHUB_PAT"),
        help="Docker Hub personal access token (default: $DOCKERHUB_PAT)"
    )
    runpod_group.add_argument(
        "--gpu-type",
        type=str,
        default=RunPodLauncher.DEFAULT_GPU_TYPE,
        help=f"GPU type to use (default: {RunPodLauncher.DEFAULT_GPU_TYPE})"
    )
    runpod_group.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)"
    )
    runpod_group.add_argument(
        "--volume-size",
        type=int,
        default=RunPodLauncher.DEFAULT_VOLUME_SIZE,
        help=f"Persistent volume size in GB (default: {RunPodLauncher.DEFAULT_VOLUME_SIZE})"
    )
    runpod_group.add_argument(
        "--container-disk-size",
        type=int,
        default=RunPodLauncher.DEFAULT_CONTAINER_DISK_SIZE,
        help=f"Container disk size in GB (default: {RunPodLauncher.DEFAULT_CONTAINER_DISK_SIZE})"
    )
    runpod_group.add_argument(
        "--image-name",
        type=str,
        default=RunPodLauncher.DEFAULT_IMAGE,
        help=f"Docker image to use (default: {RunPodLauncher.DEFAULT_IMAGE})"
    )
    runpod_group.add_argument(
        "--no-monitor",
        action="store_true",
        help="Don't monitor pod after launch"
    )
    runpod_group.add_argument(
        "--no-auto-terminate",
        action="store_true",
        help="Don't automatically terminate pod when done"
    )
    runpod_group.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Seconds between status checks (default: 60)"
    )
    runpod_group.add_argument(
        "--timeout-hours",
        type=int,
        default=24,
        help="Maximum hours before force termination (default: 24)"
    )
    
    # Training configuration (required when not using config file)
    train_group = parser.add_argument_group("Training Configuration (Required without --config)")
    train_group.add_argument(
        "--task-name",
        type=str,
        help="Unique task name for this run"
    )
    train_group.add_argument(
        "--model-type",
        choices=["efficient_unet", "siamese"],
        help="Model type to train"
    )
    train_group.add_argument(
        "--data-manifest",
        type=str,
        help="Path to dataset manifest JSON (in pod, e.g., /workspace/data/manifest.json)"
    )
    train_group.add_argument(
        "--model-dir",
        type=str,
        help="Directory for model outputs (in pod, e.g., /workspace/models/efficient_unet)"
    )
    train_group.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory for checkpoints (in pod, e.g., /workspace/models/efficient_unet/checkpoints)"
    )
    train_group.add_argument(
        "--branch",
        type=str,
        default="main",
        help="Git branch to clone (default: main)"
    )
    
    # Training parameters (optional)
    params_group = parser.add_argument_group("Training Parameters (Optional)")
    params_group.add_argument("--batch-size", type=int)
    params_group.add_argument("--train-ratio", type=float)
    params_group.add_argument("--num-workers", type=int)
    params_group.add_argument("--gaussian-sigma", type=float)
    params_group.add_argument("--weights", type=str)
    params_group.add_argument("--pretrained-model-dir", type=str)
    params_group.add_argument("--epochs", type=int)
    params_group.add_argument("--lr", type=float)
    params_group.add_argument("--epoch-vis", type=str)
    params_group.add_argument("--epoch-save-interval", type=int)
    params_group.add_argument("--device", type=str)
    params_group.add_argument("--weight-decay", type=float)
    params_group.add_argument("--activation", type=str)
    params_group.add_argument("--seed", type=int)
    params_group.add_argument("--continue-training", action="store_true")
    params_group.add_argument("--use-best-model", action="store_true")
    params_group.add_argument("--cbam", action="store_true")
    params_group.add_argument("--use-visualizer", action="store_true")
    params_group.add_argument("--autodownload", action="store_true")
    params_group.add_argument("--verbose", action="store_true")
    params_group.add_argument("--training-stages", type=str)
    params_group.add_argument("--loss-params", type=str)
    params_group.add_argument("--scheduler-type", type=str)
    params_group.add_argument("--scheduler-params", type=str)
    
    # ClearML parameters
    clearml_group = parser.add_argument_group("ClearML Configuration (Optional)")
    clearml_group.add_argument("--clearml-task", type=str, help="ClearML task name")
    clearml_group.add_argument("--clearml-project", type=str, default="aegear", help="ClearML project name")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration file if provided
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Apply config values as defaults (CLI args take precedence)
        # RunPod configuration
        if 'runpod' in config:
            runpod_cfg = config['runpod']
            if args.gpu_type == RunPodLauncher.DEFAULT_GPU_TYPE and 'gpu_type' in runpod_cfg:
                args.gpu_type = runpod_cfg['gpu_type']
            if args.gpu_count == 1 and 'gpu_count' in runpod_cfg:
                args.gpu_count = runpod_cfg['gpu_count']
            if args.volume_size == RunPodLauncher.DEFAULT_VOLUME_SIZE and 'volume_size' in runpod_cfg:
                args.volume_size = runpod_cfg['volume_size']
            if args.container_disk_size == RunPodLauncher.DEFAULT_CONTAINER_DISK_SIZE and 'container_disk_size' in runpod_cfg:
                args.container_disk_size = runpod_cfg['container_disk_size']
            if args.image_name == RunPodLauncher.DEFAULT_IMAGE and 'docker_image' in runpod_cfg:
                args.image_name = runpod_cfg['docker_image']
        
        # Training configuration
        if 'training' in config:
            train_cfg = config['training']
            for key, value in train_cfg.items():
                arg_key = key.replace('-', '_')
                cli_flag = f"--{key.replace('_', '-')}"
                if cli_flag not in sys.argv:
                    setattr(args, arg_key, value)

        # Training parameters
        if 'parameters' in config:
            params_cfg = config['parameters']
            # List of boolean flags (action="store_true" arguments)
            bool_flags = ['continue_training', 'use_best_model', 'cbam', 'use_visualizer', 
                         'autodownload', 'verbose']
            
            for key, value in params_cfg.items():
                arg_key = key.replace('-', '_')
                current_value = getattr(args, arg_key, None)
                
                # For boolean flags, check if it's False (default) and not explicitly set via CLI
                if arg_key in bool_flags:
                    # Only apply config value if current value is False (wasn't set via CLI)
                    if current_value is False:
                        setattr(args, arg_key, value)
                # For other parameters, apply if None
                elif not hasattr(args, arg_key) or current_value is None:
                    setattr(args, arg_key, value)
        
            # Training stages configuration
            if 'training_stages' in config['parameters'] and not args.training_stages:
                # Convert stages config to JSON string format that can be passed as env var
                args.training_stages = json.dumps(config['parameters']['training_stages'])
        
        # ClearML configuration
        if 'clearml' in config:
            clearml_cfg = config['clearml']
            if args.clearml_project == 'aegear' and 'project_name' in clearml_cfg:
                args.clearml_project = clearml_cfg['project_name']
        
        # Monitoring options from runpod config
        if 'runpod' in config:
            runpod_cfg = config['runpod']
            if 'monitor' in runpod_cfg and not runpod_cfg['monitor']:
                args.no_monitor = True
            if 'auto_terminate' in runpod_cfg and not runpod_cfg['auto_terminate']:
                args.no_auto_terminate = True
            if args.check_interval == 60 and 'check_interval' in runpod_cfg:
                args.check_interval = runpod_cfg['check_interval']
            if args.timeout_hours == 24 and 'timeout_hours' in runpod_cfg:
                args.timeout_hours = runpod_cfg['timeout_hours']
    

    # Validate required fields
    required_fields = ['task_name', 'model_type', 'data_manifest', 'model_dir', 'checkpoint_dir']
    missing_fields = [f for f in required_fields if not getattr(args, f, None)]
    if missing_fields:
        print(f"Error: Missing required fields: {', '.join(missing_fields)}")
        print("  Provide them via --config file or command-line arguments")
        return 1

    # Expand wildcards in task name (supports {date}, {time}, {datetime}, {timestamp})
    args.task_name = expand_task_name(args.task_name)

    # Use training task_name for ClearML task if not explicitly set
    if not args.clearml_task:
        args.clearml_task = args.task_name

    # Debug: Print boolean flags to verify they were loaded correctly
    print(f"\nBoolean flags loaded from config:")
    print(f"  autodownload: {args.autodownload}")
    print(f"  use_visualizer: {args.use_visualizer}")
    print(f"  verbose: {args.verbose}")
    print(f"  cbam: {args.cbam}")
    print(f"  continue_training: {args.continue_training}")
    print(f"  use_best_model: {args.use_best_model}\n")


    # Ensure training_stages is always a JSON string before use
    if isinstance(args.training_stages, dict):
        args.training_stages = json.dumps(args.training_stages)
    
    # Validate required credentials
    if not args.api_token:
        print("Error: RunPod API token not provided.")
        print("  Set RUNPOD_API_TOKEN environment variable or use --api-token")
        return 1

    # Initialize launcher
    launcher = RunPodLauncher(
        api_token=args.api_token,
        docker_username=args.docker_username,
        docker_pat=args.docker_pat
    )

    # Build environment variables
    env_vars = build_training_env_vars(args)

    # Display comprehensive configuration
    print("\n" + "="*80)
    print("AEGEAR TRAINING - RUNPOD LAUNCHER")
    print("="*80)
    
    # Task Information
    print("\n[TASK CONFIGURATION]")
    print("-" * 80)
    print(f"  Task Name:            {args.task_name}")
    print(f"  Model Type:           {args.model_type}")
    print(f"  Git Branch:           {args.branch}")
    print(f"  Data Manifest:        {args.data_manifest}")
    print(f"  Model Directory:      {args.model_dir}")
    print(f"  Checkpoint Directory: {args.checkpoint_dir}")
    
    # ClearML Configuration
    if args.clearml_task:
        print("\n[CLEARML TRACKING]")
        print("-" * 80)
        print(f"  Project:              {args.clearml_project}")
        print(f"  Task:                 {args.clearml_task}")
    
    # RunPod Configuration
    print("\n[RUNPOD CONFIGURATION]")
    print("-" * 80)
    print(f"  GPU Type:             {args.gpu_type}")
    print(f"  GPU Count:            {args.gpu_count}")
    print(f"  Docker Image:         {args.image_name}")
    print(f"  Volume Size:          {args.volume_size} GB")
    print(f"  Container Disk:       {args.container_disk_size} GB")
    
    # Training Parameters
    print("\n[TRAINING PARAMETERS]")
    print("-" * 80)
    
    # Parse training stages if provided to show summary
    stages_info = None
    if args.training_stages:
        try:
            # Try to parse the stages for display
            if args.training_stages.startswith(('{', '[')):
                stages_data = json.loads(args.training_stages)
                if isinstance(stages_data, dict) and 'stages' in stages_data:
                    stages_info = stages_data['stages']
                elif isinstance(stages_data, list):
                    stages_info = stages_data
        except:
            pass
    
    if stages_info:
        total_epochs = sum(stage.get('epochs', 0) for stage in stages_info)
        print(f"  Training Stages:      {len(stages_info)} stage(s)")
        print(f"  Total Epochs:         {total_epochs}")
        for idx, stage in enumerate(stages_info, 1):
            stage_name = stage.get('name', f'Stage {idx}')
            stage_epochs = stage.get('epochs', '?')
            stage_lr = stage.get('lr', '?')
            frozen = stage.get('freeze_layers', [])
            print(f"    Stage {idx} ({stage_name}): {stage_epochs} epochs, LR={stage_lr}, Frozen: {len(frozen)} layers")
    else:
        print(f"  Epochs:               {args.epochs if args.epochs else 'default'}")
        print(f"  Learning Rate:        {args.lr if args.lr else 'default'}")
    
    print(f"  Batch Size:           {args.batch_size if args.batch_size else 'default'}")
    print(f"  Weight Decay:         {args.weight_decay if args.weight_decay else 'default'}")
    print(f"  Num Workers:          {args.num_workers if args.num_workers else 'default'}")
    print(f"  Gaussian Sigma:       {args.gaussian_sigma if args.gaussian_sigma else 'default'}")
    
    # Model Configuration
    print("\n[MODEL CONFIGURATION]")
    print("-" * 80)
    print(f"  Activation:           {args.activation if args.activation else 'relu'}")
    print(f"  CBAM Attention:       {'Yes' if args.cbam else 'No'}")
    print(f"  Scheduler:            {args.scheduler_type if args.scheduler_type else 'ReduceLROnPlateau (default)'}")
    if args.loss_params:
        print(f"  Loss Parameters:      {args.loss_params}")
    
    # Options
    print("\n[OPTIONS]")
    print("-" * 80)
    print(f"  Auto-download:        {'Yes' if args.autodownload else 'No'}")
    print(f"  Use Visualizer:       {'Yes' if args.use_visualizer else 'No'}")
    print(f"  Verbose Mode:         {'Yes' if args.verbose else 'No'}")
    print(f"  Random Seed:          {args.seed if args.seed else 42}")
    
    # Monitoring
    print("\n[MONITORING]")
    print("-" * 80)
    print(f"  Monitor Pod:          {'Yes' if not args.no_monitor else 'No'}")
    print(f"  Auto Terminate:       {'Yes' if not args.no_auto_terminate else 'No'}")
    print(f"  Check Interval:       {args.check_interval}s")
    print(f"  Timeout:              {args.timeout_hours}h")
    
    print("\n" + "="*80)
    print("LAUNCHING POD...")
    print("="*80 + "\n")

    try:
        # Launch pod
        pod_id = launcher.launch_pod(
            task_name=args.task_name,
            env_vars=env_vars,
            gpu_type=args.gpu_type,
            gpu_count=args.gpu_count,
            volume_size=args.volume_size,
            container_disk_size=args.container_disk_size,
            image_name=args.image_name
        )

        print(f"\nPod launched successfully!")
        print(f"  Pod ID: {pod_id}")
        print(f"  Monitor at: https://www.runpod.io/console/pods")

        # Monitor if requested
        if not args.no_monitor:
            success = launcher.monitor_pod(
                pod_id=pod_id,
                check_interval=args.check_interval,
                timeout_hours=args.timeout_hours,
                auto_terminate=not args.no_auto_terminate
            )

            if success:
                print("\nTraining completed!")
                return 0
            else:
                print("\nTraining did not complete normally")
                return 1
        else:
            print("\nMonitoring disabled - pod will continue running")
            print(f"  Remember to terminate pod manually: {pod_id}")
            return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    """
    Example usage:
    python tools/launch_runpod_training.py --task-name efficient_unet_test_$(Get-Date -Format "yyyyMMdd_HHmmss") --branch hpo --model-type efficient_unet --data-manifest /workspace/dataset/manifest.json --model-dir /workspace/training/models/efficient_unet --checkpoint-dir /workspace/training/models/efficient_unet/checkpoints --autodownload --use-visualizer --epochs 1 --batch-size 64 --num-workers 6 --scheduler-type OneCycleLR --scheduler-params "max_lr=0.005,anneal_strategy=cos" --check-interval 30 --timeout-hours 2    """
    sys.exit(main())
