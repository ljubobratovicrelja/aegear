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

# ...existing code...
import os
import sys
import argparse
from typing import Dict
from aegear.nn.ops.runpod_launcher import RunPodLauncher
from aegear.nn.ops import build_training_env_vars



def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch RunPod training job for Aegear",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
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
    
    # Training configuration (required)
    train_group = parser.add_argument_group("Training Configuration (Required)")
    train_group.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Unique task name for this run"
    )
    train_group.add_argument(
        "--model-type",
        choices=["efficient_unet", "siamese"],
        required=True,
        help="Model type to train"
    )
    train_group.add_argument(
        "--data-manifest",
        type=str,
        required=True,
        help="Path to dataset manifest JSON (in pod, e.g., /workspace/data/manifest.json)"
    )
    train_group.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory for model outputs (in pod, e.g., /workspace/models/efficient_unet)"
    )
    train_group.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
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

    # Display configuration
    print("\n" + "="*60)
    print("AEGEAR TRAINING - RUNPOD LAUNCHER")
    print("="*60)
    print(f"Task: {args.task_name}")
    print(f"Model: {args.model_type}")
    print(f"Branch: {args.branch}")
    print(f"GPU: {args.gpu_type} x{args.gpu_count}")
    print(f"Image: {args.image_name}")
    if args.clearml_task:
        print(f"ClearML: {args.clearml_project}/{args.clearml_task}")
    print("="*60 + "\n")

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
