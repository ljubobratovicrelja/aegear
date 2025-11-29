#!/usr/bin/env python3
"""
ClearML HPO Launcher for Aegear (RunPod backend)

Creates individual ClearML tasks for each hyperparameter combination
and launches them directly on RunPod (no agents required).
Performs grid search over the specified hyperparameter space.

Can also recover results from incomplete HPO runs.

Usage:
    # Normal mode - launch new HPO:
    python clearml_runpod_hpo.py --config config/hpo_config.yaml
    
    # Recovery mode - salvage results from incomplete HPO:
    python clearml_runpod_hpo.py --recovery <orchestrator_task_id> --objective-metric <metric>
    
Example recovery:
    python clearml_runpod_hpo.py --recovery abc123def456 --objective-metric within_10px
"""

import os
import sys
import argparse
import itertools
from datetime import datetime
import yaml
from pathlib import Path
import threading
import queue

from clearml import Task

from aegear.nn.ops.runpod_launcher import RunPodLauncher
from aegear.nn.ops import build_training_env_vars
from aegear.nn.ops.exit_codes import EXIT_CUDA_UNAVAILABLE, get_exit_code_description

# Default template task configuration
DEFAULT_TEMPLATE_TASK_CONFIG = {
    "data_manifest": "/workspace/dataset/manifest.json",
    "model_dir": "/workspace/training/models/efficient_unet",
    "checkpoint_dir": "/workspace/training/models/efficient_unet/checkpoints",
    "autodownload": True,
    "use_visualizer": True,
    "device": "cuda",
}

# GPU priority fallback list (from most preferred to least preferred)
# If primary GPU is unavailable, try these in order
GPU_PRIORITY_FALLBACK = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 5090",
]

# Objective metric
METRIC_CHOICES = [
    'validation/loss',
    'avg_distance',
    'avg_confidence',
    'within_3px',
    'within_5px',
    'within_10px'
]

def load_hpo_config(config_path):
    """Load and parse HPO configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract search space
    search_space = {}
    for param_name, param_config in config['search_space'].items():
        if 'values' in param_config:
            search_space[param_name] = param_config['values']
        elif 'range' in param_config:
            # Support range specification: [min, max, step]
            min_val, max_val, step = param_config['range']
            import numpy as np
            search_space[param_name] = np.arange(min_val, max_val + step, step).tolist()
    
    return config, search_space

def extract_params_from_task_name(task_name):
    """
    Extract hyperparameters from task name.
    
    Task names follow pattern:
    HPO_lr{lr}_bs{batch_size}_wd{weight_decay}_{scheduler}_{cbam}_{activation}_gs{gaussian_sigma}_pw{pos_weight}_cw{centroid_weight}_sw{sparsity_weight}_dw{dice_weight}_{timestamp}
    
    Returns:
        dict: Extracted parameters
    """
    params = {}
    
    try:
        parts = task_name.split('_')
        
        for part in parts:
            if part.startswith('lr'):
                params['lr'] = float(part[2:])
            elif part.startswith('bs'):
                params['batch_size'] = int(part[2:])
            elif part.startswith('wd'):
                params['weight_decay'] = float(part[2:])
            elif part.startswith('gs'):
                params['gaussian_sigma'] = float(part[2:])
            elif part.startswith('pw'):
                params['pos_weight'] = float(part[2:])
            elif part.startswith('cw'):
                params['centroid_weight'] = float(part[2:])
            elif part.startswith('sw'):
                params['sparsity_weight'] = float(part[2:])
            elif part.startswith('dw'):
                params['dice_weight'] = float(part[2:])
            elif part in ['OneCycleLR', 'ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR']:
                params['scheduler_type'] = part
            elif part in ['cbam', 'nocbam']:
                params['cbam'] = (part == 'cbam')
            elif part in ['relu', 'leakyrelu', 'elu', 'gelu', 'silu', 'mish']:
                params['activation'] = part
        
        # Set defaults for any missing params
        params.setdefault('scheduler_type', 'OneCycleLR')
        params.setdefault('cbam', False)
        params.setdefault('activation', 'relu')
        params.setdefault('gaussian_sigma', 15.0)
        params.setdefault('pos_weight', 5.0)
        params.setdefault('centroid_weight', 0.0025)
        params.setdefault('sparsity_weight', 0.1)
        params.setdefault('dice_weight', 1.0)
        
    except Exception as e:
        print(f"Warning: Could not fully parse task name '{task_name}': {e}")
    
    return params

def collect_child_task_results(orchestrator_task_id, objective_metric):
    """
    Find all child tasks of the orchestrator and collect results from completed ones.
    
    Args:
        orchestrator_task_id: ClearML task ID of the orchestrator
        objective_metric: Metric name to extract from completed tasks
        
    Returns:
        tuple: (orchestrator_task, project_name, completed_results, launched_tasks)
    """
    print(f"\n{'='*60}")
    print("RECOVERY MODE - Loading orchestrator task...")
    print(f"{'='*60}\n")
    
    # Load the orchestrator task
    try:
        orchestrator_task = Task.get_task(task_id=orchestrator_task_id)
        project_name = orchestrator_task.get_project_name()
        print(f" Found orchestrator: {orchestrator_task.name}")
        print(f"  Project: {project_name}")
        print(f"  Created: {orchestrator_task.data.created}")
        print(f"  Status: {orchestrator_task.get_status()}")
    except Exception as e:
        print(f"âœ— Failed to load orchestrator task: {e}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Extracting launched task names from orchestrator logs...")
    print(f"{'='*60}\n")
    
    # Parse the orchestrator's console output to get the exact list of launched tasks
    launched_task_names = []
    try:
        import re
        from clearml.backend_api.session.client import APIClient
        
        print("Fetching console logs from orchestrator task...")
        
        # Get the console output using the backend API client
        client = APIClient()
        
        # Fetch logs in batches (max 10000 per request)
        all_events = []
        batch_size = 10000
        order = 'asc'  # Start from beginning
        
        while True:
            events_response = client.events.get_task_log(
                task=orchestrator_task_id,
                batch_size=batch_size,
                order=order
            )
            
            if hasattr(events_response, 'events') and events_response.events:
                batch_count = len(events_response.events)
                all_events.extend(events_response.events)
                
                if batch_count < batch_size:
                    # Got fewer events than requested, we're done
                    break
                    
                # If we got a full batch, there might be more
                # Use the timestamp of the last event to get the next batch
                if hasattr(events_response.events[-1], 'timestamp'):
                    # This might need adjustment based on API capabilities
                    break  # For now, assume we got everything
                else:
                    break
            else:
                break
        
        print(f"Retrieved {len(all_events)} log events")
        
        # Create a response-like object with all events
        class EventsContainer:
            def __init__(self, events):
                self.events = events
        
        events_response = EventsContainer(all_events)
        
        # Parse console output for task names
        console_text = ""
        
        if hasattr(events_response, 'events') and events_response.events:
            for event in events_response.events:
                # Try different field names
                log_line = None
                
                # Try to get the log message from various possible fields
                for field in ['msg', 'message', 'text', 'line', 'data']:
                    if hasattr(event, field):
                        val = getattr(event, field)
                        if val:
                            log_line = str(val)
                            break
                
                # If event is a dict
                if log_line is None and isinstance(event, dict):
                    for key in ['msg', 'message', 'text', 'line', 'data']:
                        if key in event and event[key]:
                            log_line = str(event[key])
                            break
                
                if log_line:
                    console_text += log_line + "\n"
        
        # Pattern to match task names in the logs
        # Looking for lines like: "Creating task: HPO_lr..." or "- HPO_lr..."
        # The task name format: HPO_lr{lr}_bs{batch}_wd{wd}_{scheduler}_{cbam}_{activation}_gs{sigma}_pw{pw}_cw{cw}_sw{sw}_dw{dw}_{timestamp}
        task_name_pattern = r'HPO_lr[\d.]+_bs\d+_wd[\d.e-]+[^\s\n]*'
        matches = re.findall(task_name_pattern, console_text)
        
        # Get unique task names
        launched_task_names = list(set(matches))
        
        if launched_task_names:
            print(f"[OK] Extracted {len(launched_task_names)} unique task names from orchestrator logs")
        else:
            print("[ERROR] No task names found in orchestrator logs!")
            print("Cannot proceed with recovery without task list.")
            sys.exit(1)
            
    except Exception as e:
        print(f"[WARNING] Could not parse orchestrator logs: {e}")
        import traceback
        traceback.print_exc()
        print("\n[ERROR] Cannot proceed without parsing orchestrator logs.")
        sys.exit(1)
    
    # Verify we have task names
    if not launched_task_names:
        print("[ERROR] No task names extracted. Cannot proceed.")
        sys.exit(1)
    
    # Get the tasks by name
    print(f"\nLooking up {len(launched_task_names)} tasks from orchestrator logs...\n")
    try:
        hpo_tasks = []
        
        from tqdm import tqdm
        for i, task_name in enumerate(tqdm(launched_task_names, desc="Loading HPO tasks"), 1):
            try:
                tasks = Task.get_tasks(
                    project_name=project_name,
                    task_name=task_name,
                    task_filter={'status': ['created', 'in_progress', 'queued', 'completed', 'failed', 'stopped']}
                )
                if tasks:
                    # Get the most recent task with this name (in case of retries)
                    task = max(tasks, key=lambda t: t.data.created)
                    hpo_tasks.append(task)
            except Exception as e:
                print(f"  ⚠ Error loading task {task_name}: {e}")
        
        print(f"\n✓ Successfully loaded {len(hpo_tasks)}/{len(launched_task_names)} tasks\n")
        
        # Collect results from all tasks
        completed_results = []
        launched_tasks = []  # Reconstruct for compatibility with report generation
        
        for task in tqdm(hpo_tasks, desc="Collecting task results"):
            status = task.get_status()
            task_name = task.name
            
            print(f"Processing: {task_name}")
            print(f"  Status: {status}")
            
            # Extract parameters from task name
            params = extract_params_from_task_name(task_name)
            
            # Add to launched_tasks for report compatibility
            launched_tasks.append({
                'task_name': task_name,
                'pod_id': 'recovered',
                'params': params,
                'launch_time': task.data.created.timestamp(),
                'job_config': {},
                'retry_count': 0,
                'gpu_type': 'recovered'
            })
            
            result = {
                'task_name': task_name,
                'task_id': task.id,
                'pod_id': 'recovered',
                'params': params,
                'status': status,
                'metrics': {}
            }
            
            if status == 'completed':
                # Try to get metrics
                try:
                    metrics = task.get_last_scalar_metrics()
                    
                    # Extract metrics from Summary section
                    if 'Summary' in metrics:
                        summary = metrics['Summary']
                        
                        # Get the objective metric
                        final_metric_key = f'final_{objective_metric}'
                        if final_metric_key in summary:
                            result['metrics'][objective_metric] = summary[final_metric_key]['last']
                            print(f"  {objective_metric}: {result['metrics'][objective_metric]:.4f}")
                        
                        # Collect ALL summary metrics
                        for metric_key, metric_data in summary.items():
                            if metric_key.startswith('final_') and isinstance(metric_data, dict):
                                clean_name = metric_key.replace('final_', '')
                                if clean_name not in result['metrics']:
                                    result['metrics'][clean_name] = metric_data['last']
                    
                    # Also get validation loss from 'loss' section
                    if 'loss' in metrics and 'val' in metrics['loss']:
                        result['metrics']['validation/loss'] = metrics['loss']['val']['last']
                        
                except Exception as e:
                    print(f"  Could not retrieve metrics: {e}")
            else:
                print(f"  Task not completed, skipping metrics")
            
            completed_results.append(result)
        
        print(f"\n{'='*60}")
        print(f"Recovery Summary:")
        completed_count = sum(1 for r in completed_results if r['status'] == 'completed')
        with_metrics = sum(1 for r in completed_results 
                          if r['status'] == 'completed' and objective_metric in r['metrics'])
        print(f"  Total HPO tasks found: {len(hpo_tasks)}")
        print(f"  Completed tasks: {completed_count}")
        print(f"  With {objective_metric} metric: {with_metrics}")
        print(f"{'='*60}\n")
        
        return orchestrator_task, project_name, completed_results, launched_tasks
        
    except Exception as e:
        print(f" Error searching for child tasks: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def keyboard_listener(quit_queue):
    """
    Thread function to listen for 'q' key press to quit HPO.
    
    Args:
        quit_queue: Queue to signal quit request
    """
    try:
        while True:
            user_input = input()
            if user_input.strip().lower() == 'q':
                quit_queue.put('quit')
                break
    except (EOFError, KeyboardInterrupt):
        # Handle case where input stream is closed or interrupted
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="ClearML HPO Launcher for Aegear (RunPod backend)")
    parser.add_argument('--config', type=str, default='config/hpo_config.yaml',
                        help='Path to HPO configuration YAML file (default: config/hpo_config.yaml)')
    parser.add_argument('--recovery', type=str, default=None,
                        help='Recovery mode: provide orchestrator task ID to recover results from incomplete HPO')
    parser.add_argument('--objective-metric', type=str, choices=METRIC_CHOICES, default=None,
                        help='Metric to optimize (overrides config file, required in recovery mode)')
    parser.add_argument('--epochs', type=int, default=None, 
                        help='Number of training epochs (overrides config file)')
    parser.add_argument('--num-workers', type=int, default=None, 
                        help='Number of data loader workers (overrides config file)')
    parser.add_argument('--max-tests', type=int, default=None,
                        help='Maximum number of HPO experiments to run (overrides config file)')
    return parser.parse_args()

# --- RunPod credentials ---
RUNPOD_API_TOKEN = os.getenv("RUNPOD_API_TOKEN")
DOCKERHUB_USERNAME = os.getenv("DOCKERHUB_USERNAME")
DOCKERHUB_PAT = os.getenv("DOCKERHUB_PAT")

if not RUNPOD_API_TOKEN:
    print("Error: RUNPOD_API_TOKEN not set.")
    sys.exit(1)


def launch_pod_with_fallback(launcher, task_name, env_vars, primary_gpu_type, gpu_count, 
                             volume_size, container_disk_size, docker_image):
    """
    Launch a pod with GPU fallback logic.
    
    Tries to launch with the primary GPU type first. If unavailable,
    falls back to other GPU types in priority order.
    
    Args:
        launcher: RunPodLauncher instance
        task_name: Name for the pod
        env_vars: Environment variables dictionary
        primary_gpu_type: Preferred GPU type
        gpu_count: Number of GPUs
        volume_size: Volume size in GB
        container_disk_size: Container disk size in GB
        docker_image: Docker image name
        
    Returns:
        tuple: (pod_id, gpu_type_used) if successful
        
    Raises:
        Exception: If all GPU types fail
    """
    # Build priority list starting with primary GPU
    gpu_priority_list = [primary_gpu_type]
    
    # Add fallback GPUs that aren't already in the list
    for fallback_gpu in GPU_PRIORITY_FALLBACK:
        if fallback_gpu != primary_gpu_type:
            gpu_priority_list.append(fallback_gpu)
    
    last_error = None
    
    for i, gpu_type in enumerate(gpu_priority_list):
        try:
            print(f"\n{'  ' if i > 0 else ''}Attempting to launch with GPU: {gpu_type}")
            
            pod_id = launcher.launch_pod(
                task_name=task_name,
                env_vars=env_vars,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                volume_size=volume_size,
                container_disk_size=container_disk_size,
                image_name=docker_image
            )
            
            # Success!
            if i > 0:
                print(f"  âœ“ Successfully launched with fallback GPU: {gpu_type}")
            
            return pod_id, gpu_type
            
        except Exception as e:
            error_msg = str(e).lower()
            last_error = e
            
            # Check if error is due to GPU unavailability
            # Common RunPod error patterns for resource unavailability
            is_gpu_unavailable = (
                "no longer available" in error_msg or
                "not available" in error_msg or
                "no capacity" in error_msg or
                "unavailable" in error_msg or
                "could not find" in error_msg or
                "does not have the resources" in error_msg or
                "no resources" in error_msg or
                "please try a different machine" in error_msg
            )
            
            if is_gpu_unavailable:
                print(f"  ⚠ GPU type '{gpu_type}' is not available")
                
                # Try next GPU in priority list
                if i < len(gpu_priority_list) - 1:
                    next_gpu = gpu_priority_list[i + 1]
                    print(f"  → Falling back to: {next_gpu}")
                    continue
                else:
                    print(f"  ✗ All GPU types exhausted")
                    raise RuntimeError(
                        f"Failed to launch pod - all GPU types unavailable. "
                        f"Tried: {', '.join(gpu_priority_list)}"
                    ) from last_error
            else:
                # Different error - don't try fallback
                print(f"  ✗ Failed with non-availability error: {str(e)}")
                raise
    
    # Should never reach here, but just in case
    raise RuntimeError(
        f"Failed to launch pod after trying all GPU types: {gpu_priority_list}"
    ) from last_error


def main():
    args = parse_args()
    
    # Check if we're in recovery mode
    if args.recovery:
        if not args.objective_metric:
            print("Error: --objective-metric is required in recovery mode")
            sys.exit(1)
        
        # Recovery mode: collect results from existing orchestrator
        original_orchestrator, project_name, completed_results, launched_tasks = collect_child_task_results(
            args.recovery,
            args.objective_metric
        )
        
        # Create a new orchestrator task for the recovery
        print(f"\n{'='*60}")
        print("Creating recovery orchestrator task...")
        print(f"{'='*60}\n")
        
        orchestrator_task = Task.init(
            project_name=project_name,
            task_name=f"HPO Recovery - {original_orchestrator.name} - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type=Task.TaskTypes.optimizer,
            reuse_last_task_id=False
        )
        
        # Add reference to original orchestrator
        orchestrator_task.set_comment(f"""
Recovered HPO results from incomplete orchestrator run.

Original Orchestrator:
- Task ID: {args.recovery}
- Task Name: {original_orchestrator.name}
- Created: {original_orchestrator.data.created}
- Status: {original_orchestrator.get_status()}

Recovery Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This task contains compiled results from all completed child tasks.
""")
        
        print(f" Created recovery orchestrator: {orchestrator_task.name}")
        print(f"  Task ID: {orchestrator_task.id}")
        print(f"  Original orchestrator: {args.recovery}")
        
        objective_metric = args.objective_metric
        
        # Skip to report generation (reuse the existing report code)
        # The completed_results and launched_tasks are now populated from recovery
        
    else:
        # Normal mode: launch new HPO
        # Load configuration from YAML file
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        
        print(f"Loading configuration from: {config_path}")
        config, search_space = load_hpo_config(config_path)
    
        # Extract configuration values
        project_name = config['clearml']['project_name']
        gpu_type = config['runpod']['gpu_type']
        docker_image = config['runpod']['docker_image']
        volume_size = config['runpod']['volume_size']
        container_disk_size = config['runpod']['container_disk_size']
        
        # Training configuration (allow CLI overrides)
        epochs = args.epochs if args.epochs is not None else config['training']['epochs']
        num_workers = args.num_workers if args.num_workers is not None else config['training']['num_workers']
        model_type = config['training']['model_type']
        branch = config['training']['branch']
        
        # Optimization settings (allow CLI overrides)
        objective_metric = args.objective_metric if args.objective_metric else config['optimization']['objective_metric']
        max_tests = args.max_tests if args.max_tests is not None else config['optimization']['max_tests']
        
        print("="*60)
        print("ClearML HPO with RunPod Backend")
        print("="*60)
        print(f"\nObjective metric: {objective_metric}")
        print(f"Search space:")
        for param, values in search_space.items():
            print(f"  - {param}: {values}")
        
        # Create orchestrator task
        orchestrator_task = Task.init(
            project_name=project_name,
            task_name=f"HPO Orchestrator - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_type=Task.TaskTypes.optimizer,
            reuse_last_task_id=False
        )
    
        # Initialize RunPod launcher
        launcher = RunPodLauncher(
            api_token=RUNPOD_API_TOKEN,
            docker_username=DOCKERHUB_USERNAME,
            docker_pat=DOCKERHUB_PAT
        )
        
        # Generate all parameter combinations (grid search)
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        combinations = list(itertools.product(*param_values))
        
        # Limit number of tests if specified
        if max_tests and max_tests < len(combinations):
            import random
            combinations = random.sample(combinations, max_tests)
            print(f"\nLimited to {max_tests} random tests from {len(list(itertools.product(*param_values)))} total combinations")
        
        total_jobs = len(combinations)
        print(f"\nTotal tests to run: {total_jobs}")
        print(f"GPU type: {gpu_type}\n")
        
        launched_tasks = []
        retry_configs = []  # Track configurations that need retry due to machine issues
        max_retries_per_config = 3  # Maximum retry attempts for CUDA unavailable errors
        
        for i, combo in enumerate(combinations, 1):
            # Create parameter dict for this combination
            params = dict(zip(param_names, combo))
            lr = params['lr']
            batch_size = params['batch_size']
            weight_decay = params['weight_decay']
            scheduler_type = params.get('scheduler_type', 'OneCycleLR')
            cbam = params.get('cbam', False)
            activation = params.get('activation', 'relu')
            gaussian_sigma = params.get('gaussian_sigma', 15.0)
            
            # Extract loss weight parameters
            pos_weight = params.get('pos_weight', 5.0)
            centroid_weight = params.get('centroid_weight', 0.0025)
            sparsity_weight = params.get('sparsity_weight', 0.1)
            dice_weight = params.get('dice_weight', 1.0)
            
            # Construct loss_params string
            loss_params = f"pos_weight={pos_weight},centroid_weight={centroid_weight},sparsity_weight={sparsity_weight},dice_weight={dice_weight}"
            
            # Configure scheduler parameters based on type
            if scheduler_type == 'OneCycleLR':
                scheduler_params = f"max_lr={lr},anneal_strategy=cos"
            elif scheduler_type == 'ReduceLROnPlateau':
                scheduler_params = f"mode=min,factor=0.5,patience=3,min_lr=1e-6"
            else:
                scheduler_params = ""
            
            # Create a unique task name with timestamp to avoid collision with archived tasks
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cbam_str = "cbam" if cbam else "nocbam"
            task_name = f"HPO_lr{lr}_bs{batch_size}_wd{weight_decay}_{scheduler_type}_{cbam_str}_{activation}_gs{gaussian_sigma}_pw{pos_weight}_cw{centroid_weight}_sw{sparsity_weight}_dw{dice_weight}_{timestamp}"
            
            print(f"[{i}/{total_jobs}] Creating task: {task_name}")
            print(f"  Parameters: lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}, scheduler={scheduler_type}, cbam={cbam}, activation={activation}, gaussian_sigma={gaussian_sigma}")
            print(f"  Loss weights: pos_weight={pos_weight}, centroid_weight={centroid_weight}, sparsity_weight={sparsity_weight}, dice_weight={dice_weight}")
            
            # Build job configuration
            # Note: Don't create ClearML task here - let the training script create it
            # This ensures proper lifecycle management and console output capture
            job_config = DEFAULT_TEMPLATE_TASK_CONFIG.copy()
            job_config.update({
                "branch": branch,
                "model_type": model_type,
                "lr": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "num_workers": num_workers,
                "scheduler_type": scheduler_type,
                "scheduler_params": scheduler_params,
                "task_name": task_name,
                "clearml_task": task_name,
                "clearml_project": project_name,
                # Add all required attributes for build_training_env_vars
                "train_ratio": None,
                "gaussian_sigma": gaussian_sigma,
                "weights": None,
                "pretrained_model_dir": None,
                "epoch_vis": None,
                "epoch_save_interval": None,
                "activation": activation,
                "seed": None,
                "continue_training": False,
                "use_best_model": False,
                "cbam": cbam,
                "verbose": False,
                "training_stages": None,
                "loss_params": loss_params,
            })
            
            # Build environment variables
            args_ns = argparse.Namespace(**job_config)
            env_vars = build_training_env_vars(args_ns)
            
            # Training script will create task via Task.init() with these values
            # This ensures proper console capture and lifecycle management
            env_vars["CLEARML_TASK"] = task_name
            env_vars["CLEARML_PROJECT"] = project_name
            
            try:
                # Launch on RunPod with GPU fallback
                import time
                pod_id, gpu_used = launch_pod_with_fallback(
                    launcher=launcher,
                    task_name=task_name,
                    env_vars=env_vars,
                    primary_gpu_type=gpu_type,
                    gpu_count=1,
                    volume_size=volume_size,
                    container_disk_size=container_disk_size,
                    docker_image=docker_image
                )
                
                print(f"  Pod launched: {pod_id}")
                if gpu_used != gpu_type:
                    print(f"    Note: Using fallback GPU '{gpu_used}' instead of '{gpu_type}'")
                print(f"    Task will be created as: {task_name}")
                
                launched_tasks.append({
                    'task_name': task_name,
                    'pod_id': pod_id,
                    'params': params,
                    'launch_time': time.time(),
                    'job_config': job_config.copy(),
                    'retry_count': 0,
                    'gpu_type': gpu_used  # Track which GPU was actually used
                })
                
            except Exception as e:
                print(f"  âœ— Failed to launch pod: {e}")
                # Note: No individual task to mark as failed since it doesn't exist yet
        
        # Log initial summary to orchestrator task
        summary = f"""
    HPO Launch Summary
    ==================
    Total tests: {total_jobs}
    Successfully launched: {len(launched_tasks)}
    Primary GPU requested: {gpu_type}
    
    Launched Tasks:
    """
        for task_info in launched_tasks:
            summary += f"\n- {task_info['task_name']}"
            summary += f"\n  Pod ID: {task_info['pod_id']}"
            summary += f"\n  GPU: {task_info.get('gpu_type', 'unknown')}"
            summary += f"\n  Params: {task_info['params']}"
        
        orchestrator_task.get_logger().report_text(summary)
        
        print("\n" + "="*60)
        print("All pods launched successfully!")
        print(f"Successfully launched: {len(launched_tasks)}/{total_jobs} pods")
        print(f"Monitor pods at: https://www.runpod.io/console/pods")
        print(f"Monitor ClearML at: {orchestrator_task.get_output_log_web_page()}")
        print("="*60)
    
    # Now monitor task completion and collect results (common to both normal and recovery modes)
    # In recovery mode, this section is skipped as we already have completed_results
    if not args.recovery:
        print(f"\n{'='*60}")
        print("Monitoring task completion...")
        print(f"{'='*60}")
        print("\nPress 'q' + Enter to quit HPO early and compile results from completed tasks.\n")
        
        import time
        completed_results = []
        check_interval = 5  # Check every 5 seconds
        
        # Set up keyboard listener for graceful quit
        quit_queue = queue.Queue()
        keyboard_thread = threading.Thread(target=keyboard_listener, args=(quit_queue,), daemon=True)
        keyboard_thread.start()
        quit_requested = False
        
        def sync_task_status():
            """Sync all task statuses and update completed_results. Returns updated counts."""
            from clearml import Task as TaskQuery
            newly_completed = 0
            
            for task_info in launched_tasks:
                # Skip if already collected
                if any(r['task_name'] == task_info['task_name'] for r in completed_results):
                    continue
                
                # Try to find the task by name
                try:
                    tasks = TaskQuery.get_tasks(
                        project_name=project_name,
                        task_name=task_info['task_name'],
                        task_filter={'status': ['created', 'in_progress', 'queued', 'completed', 'failed', 'stopped']}
                    )
                    
                    if not tasks:
                        continue
                        
                    task = tasks[0] if len(tasks) == 1 else max(tasks, key=lambda t: t.data.created)
                    task_created_time = task.data.created.timestamp()
                    if task_created_time < task_info['launch_time']:
                        continue
                        
                    status = task.get_status()
                except Exception as e:
                    continue
                
                if status in ['completed', 'failed', 'stopped', 'aborted']:
                    exit_code = launcher.get_pod_exit_code(task_info['pod_id'])
                    
                    if exit_code == EXIT_CUDA_UNAVAILABLE and task_info['retry_count'] < max_retries_per_config:
                        # Only print to console, not to ClearML logs (will be noisy)
                        sys.stdout.write(f"\n Task failed due to CUDA unavailability: {task_info['task_name']}\n")
                        sys.stdout.write(f"  Exit code: {exit_code} ({get_exit_code_description(exit_code)})\n")
                        sys.stdout.write(f"  Retry attempt {task_info['retry_count'] + 1}/{max_retries_per_config}\n")
                        sys.stdout.write(f"  Will retry this configuration on a different machine...\n")
                        sys.stdout.flush()
                        
                        retry_configs.append({
                            'task_info': task_info,
                            'reason': 'CUDA unavailable'
                        })
                        
                        completed_results.append({
                            'task_name': task_info['task_name'],
                            'task_id': task.id if task else None,
                            'pod_id': task_info['pod_id'],
                            'params': task_info['params'],
                            'status': 'retry_scheduled',
                            'exit_code': exit_code,
                            'metrics': {}
                        })
                        newly_completed += 1
                        continue
                    
                    # Print completion notification
                    sys.stdout.write(f"\n Task completed: {task_info['task_name']} (status: {status}, exit_code: {exit_code})\n")
                    sys.stdout.flush()
                    
                    result = {
                        'task_name': task_info['task_name'],
                        'task_id': task.id,
                        'pod_id': task_info['pod_id'],
                        'params': task_info['params'],
                        'status': status,
                        'exit_code': exit_code,
                        'metrics': {}
                    }
                    
                    if status == 'completed':
                        try:
                            metrics = task.get_last_scalar_metrics()
                            
                            if 'Summary' in metrics:
                                summary = metrics['Summary']
                                final_metric_key = f'final_{objective_metric}'
                                if final_metric_key in summary:
                                    result['metrics'][objective_metric] = summary[final_metric_key]['last']
                                    sys.stdout.write(f"  {objective_metric}: {result['metrics'][objective_metric]:.4f}\n")
                                    sys.stdout.flush()
                                
                                for metric_key, metric_data in summary.items():
                                    if metric_key.startswith('final_') and isinstance(metric_data, dict):
                                        clean_name = metric_key.replace('final_', '')
                                        if clean_name not in result['metrics']:
                                            result['metrics'][clean_name] = metric_data['last']
                                
                                sys.stdout.write(f"  Collected all summary metrics: {list(result['metrics'].keys())}\n")
                                sys.stdout.flush()
                            
                            if 'loss' in metrics and 'val' in metrics['loss']:
                                result['metrics']['validation/loss'] = metrics['loss']['val']['last']
                        except Exception as e:
                            sys.stdout.write(f"  Could not retrieve metrics for {task_info['task_name']}: {e}\n")
                            sys.stdout.flush()
                    
                    completed_results.append(result)
                    newly_completed += 1
            
            return newly_completed
        
        while len(completed_results) < len(launched_tasks):
            # Sync task statuses
            newly_completed = sync_task_status()
            
            # Check if all tasks are done or quit requested
            if len(completed_results) < len(launched_tasks):
                remaining = len(launched_tasks) - len(completed_results)
                # Use \r for carriage return to keep updating same line (won't clutter ClearML logs)
                sys.stdout.write(f"\r[{datetime.now().strftime('%H:%M:%S')}] Completed: {len(completed_results)}/{len(launched_tasks)} | Remaining: {remaining} | Next check in {check_interval}s...")
                sys.stdout.flush()
                
                # Check for quit signal with timeout
                try:
                    quit_signal = quit_queue.get(timeout=check_interval)
                    if quit_signal == 'quit':
                        quit_requested = True
                        break
                except queue.Empty:
                    # No quit signal, continue monitoring
                    pass
        
        sys.stdout.write('\n')  # Newline after monitoring loop
        sys.stdout.flush()
        
        # Handle graceful quit if requested
        if quit_requested:
            # Force a final sync before showing confirmation to get accurate counts
            print("\nSyncing task statuses before quit...")
            sync_task_status()
            
            print(f"\n{'='*60}")
            print("QUIT REQUESTED")
            print(f"{'='*60}\n")
            
            n_running = len(launched_tasks) - len(completed_results)
            n_completed = len([r for r in completed_results if r['status'] == 'completed' and objective_metric in r['metrics']])
            
            if n_completed == 0:
                print(f"Quitting HPO - this will stop {n_running} still ongoing tasks.")
                print("No completed tasks with valid metrics found - HPO will shut down without compiling results.")
            else:
                print(f"Quitting HPO - this will stop {n_running} still ongoing tasks and compile the HPO with {n_completed} completed tasks.")
            
            print("\nType 'agree' to proceed: ", end='', flush=True)
            try:
                confirmation = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                confirmation = ''
            
            if confirmation == 'agree':
                print(f"\n{'='*60}")
                print("Shutting down HPO...")
                print(f"{'='*60}\n")
                
                # Stop all running ClearML tasks and terminate all RunPod pods
                from clearml import Task as TaskQuery
                stopped_task_count = 0
                terminated_pod_count = 0
                
                for task_info in launched_tasks:
                    # Skip tasks we've already collected
                    if any(r['task_name'] == task_info['task_name'] for r in completed_results):
                        continue
                    
                    # Try to stop ClearML task if it exists
                    clearml_task_stopped = False
                    try:
                        # Find the task
                        tasks = TaskQuery.get_tasks(
                            project_name=project_name,
                            task_name=task_info['task_name'],
                            task_filter={'status': ['created', 'in_progress', 'queued']}
                        )
                        
                        if tasks:
                            task = tasks[0] if len(tasks) == 1 else max(tasks, key=lambda t: t.data.created)
                            task_created_time = task.data.created.timestamp()
                            
                            # Verify this is the task we launched
                            if task_created_time >= task_info['launch_time']:
                                print(f"Stopping ClearML task: {task_info['task_name']}")
                                task.mark_stopped()
                                stopped_task_count += 1
                                clearml_task_stopped = True
                    except Exception as e:
                        print(f"Warning: Could not stop ClearML task {task_info['task_name']}: {e}")
                    
                    # Always try to terminate the RunPod pod, even if ClearML task doesn't exist yet
                    try:
                        pod_id = task_info['pod_id']
                        if not clearml_task_stopped:
                            print(f"Terminating pod for {task_info['task_name']}: {pod_id}")
                        else:
                            print(f"  Terminating pod: {pod_id}")
                        
                        launcher.terminate_pod(pod_id)
                        terminated_pod_count += 1
                    except Exception as e:
                        print(f"  Warning: Could not terminate pod {pod_id}: {e}")
                
                print(f"\nStopped {stopped_task_count} ClearML tasks and terminated {terminated_pod_count} RunPod pods.")
                
                if n_completed == 0:
                    # No completed tasks - just shut down orchestrator and exit
                    print("\nNo completed tasks to compile. Shutting down orchestrator.")
                    orchestrator_task.mark_stopped()
                    orchestrator_task.get_logger().report_text(
                        "HPO manually stopped with no completed tasks. No results compiled.",
                        print_console=False
                    )
                    print(f"\n{'='*60}")
                    print("HPO shut down.")
                    print(f"{'='*60}")
                    sys.exit(0)
                
                # Continue to results compilation with completed tasks
                print(f"\nProceeding to compile results from {n_completed} completed tasks...\n")
            else:
                print("\nQuit cancelled. Continuing HPO monitoring...\n")
                quit_requested = False
                # Continue monitoring
                # (This will fall through to the retry logic if needed)
    
        # Process retries for configurations that failed due to machine issues
        if retry_configs and not quit_requested:
            print(f"\n{'='*60}")
            print(f"RETRYING FAILED CONFIGURATIONS")
            print(f"{'='*60}")
            print(f"Found {len(retry_configs)} configurations to retry due to machine issues\n")
            
            for retry_info in retry_configs:
                original_task = retry_info['task_info']
                retry_count = original_task['retry_count'] + 1
                
                # Create new task name with retry suffix
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                params = original_task['params']
                lr = params['lr']
                batch_size = params['batch_size']
                weight_decay = params['weight_decay']
                scheduler_type = params.get('scheduler_type', 'OneCycleLR')
                cbam = params.get('cbam', False)
                activation = params.get('activation', 'relu')
                gaussian_sigma = params.get('gaussian_sigma', 15.0)
                cbam_str = "cbam" if cbam else "nocbam"
                
                task_name = f"HPO_lr{lr}_bs{batch_size}_wd{weight_decay}_{scheduler_type}_{cbam_str}_{activation}_gs{gaussian_sigma}_retry{retry_count}_{timestamp}"
                
                print(f"Retrying configuration (attempt {retry_count}):")
                print(f"  Original task: {original_task['task_name']}")
                print(f"  New task: {task_name}")
                print(f"  Reason: {retry_info['reason']}")
                
                # Use the original job config but update task name
                job_config = original_task['job_config'].copy()
                job_config['task_name'] = task_name
                job_config['clearml_task'] = task_name
                
                # Rebuild environment variables
                args_ns = argparse.Namespace(**job_config)
                env_vars = build_training_env_vars(args_ns)
                env_vars["CLEARML_TASK"] = task_name
                env_vars["CLEARML_PROJECT"] = project_name
                
                try:
                    # Launch on RunPod with GPU fallback
                    import time
                    pod_id, gpu_used = launch_pod_with_fallback(
                        launcher=launcher,
                        task_name=task_name,
                        env_vars=env_vars,
                        primary_gpu_type=gpu_type,
                        gpu_count=1,
                        volume_size=volume_size,
                        container_disk_size=container_disk_size,
                        docker_image=docker_image
                    )
                    
                    print(f"  Retry pod launched: {pod_id}")
                    if gpu_used != gpu_type:
                        print(f"    Note: Using fallback GPU '{gpu_used}' instead of '{gpu_type}'")
                    print()
                    
                    launched_tasks.append({
                        'task_name': task_name,
                        'pod_id': pod_id,
                        'params': params,
                        'launch_time': time.time(),
                        'job_config': job_config.copy(),
                        'retry_count': retry_count,
                        'gpu_type': gpu_used  # Track which GPU was actually used
                    })
                    
                except Exception as e:
                    print(f"  Failed to launch retry pod: {e}\n")
            
            # Monitor retry tasks
            if any(t['retry_count'] > 0 for t in launched_tasks):
                print(f"\n{'='*60}")
                print("Monitoring retry tasks...")
                print(f"{'='*60}\n")
                
                # Clear completed_results to restart monitoring
                retry_task_names = [t['task_name'] for t in launched_tasks if t['retry_count'] > 0]
                completed_results = [r for r in completed_results if r['task_name'] not in retry_task_names]
                
                # Run monitoring loop again for retry tasks (skip if quit was requested)
                # Use same check_interval (5 seconds) for retry monitoring
                while len(completed_results) < len(launched_tasks) and not quit_requested:
                    for task_info in launched_tasks:
                        # Skip if already collected
                        if any(r['task_name'] == task_info['task_name'] for r in completed_results):
                            continue
                        
                        try:
                            from clearml import Task as TaskQuery
                            tasks = TaskQuery.get_tasks(
                                project_name=project_name,
                                task_name=task_info['task_name'],
                                task_filter={'status': ['created', 'in_progress', 'queued', 'completed', 'failed', 'stopped']}
                            )
                            
                            if not tasks:
                                continue
                                
                            task = tasks[0] if len(tasks) == 1 else max(tasks, key=lambda t: t.data.created)
                            task_created_time = task.data.created.timestamp()
                            if task_created_time < task_info['launch_time']:
                                continue
                                
                            status = task.get_status()
                        except Exception as e:
                            continue
                        
                        if status in ['completed', 'failed', 'stopped', 'aborted']:
                            exit_code = launcher.get_pod_exit_code(task_info['pod_id'])
                            
                            if exit_code == EXIT_CUDA_UNAVAILABLE and task_info['retry_count'] < max_retries_per_config:
                                print(f" Retry task also failed due to CUDA: {task_info['task_name']}")
                                print(f"  Will retry again (attempt {task_info['retry_count'] + 1}/{max_retries_per_config})")
                                
                                retry_configs.append({
                                    'task_info': task_info,
                                    'reason': 'CUDA unavailable (retry)'
                                })
                                
                                completed_results.append({
                                    'task_name': task_info['task_name'],
                                    'task_id': task.id if task else None,
                                    'pod_id': task_info['pod_id'],
                                    'params': task_info['params'],
                                    'status': 'retry_scheduled',
                                    'exit_code': exit_code,
                                    'metrics': {}
                                })
                                continue
                            
                            print(f" Retry task completed: {task_info['task_name']} (status: {status}, exit_code: {exit_code})")
                            
                            result = {
                                'task_name': task_info['task_name'],
                                'task_id': task.id,
                                'pod_id': task_info['pod_id'],
                                'params': task_info['params'],
                                'status': status,
                                'exit_code': exit_code,
                                'metrics': {}
                            }
                            
                            if status == 'completed':
                                try:
                                    metrics = task.get_last_scalar_metrics()
                                    if 'Summary' in metrics:
                                        summary = metrics['Summary']
                                        final_metric_key = f'final_{objective_metric}'
                                        if final_metric_key in summary:
                                            result['metrics'][objective_metric] = summary[final_metric_key]['last']
                                            print(f"  {objective_metric}: {result['metrics'][objective_metric]:.4f}")
                                        
                                        for metric_key, metric_data in summary.items():
                                            if metric_key.startswith('final_') and isinstance(metric_data, dict):
                                                clean_name = metric_key.replace('final_', '')
                                                if clean_name not in result['metrics']:
                                                    result['metrics'][clean_name] = metric_data['last']
                                    
                                    if 'loss' in metrics and 'val' in metrics['loss']:
                                        result['metrics']['validation/loss'] = metrics['loss']['val']['last']
                                except Exception as e:
                                    print(f"   Could not retrieve metrics: {e}")
                            
                            completed_results.append(result)
                    
                    if len(completed_results) < len(launched_tasks):
                        remaining = len(launched_tasks) - len(completed_results)
                        # Use \r for carriage return to keep updating same line
                        sys.stdout.write(f"\r[{datetime.now().strftime('%H:%M:%S')}] Retry monitoring - Completed: {len(completed_results)}/{len(launched_tasks)} | Remaining: {remaining} | Next check in {check_interval}s...")
                        sys.stdout.flush()
                        
                        # Check for quit signal with timeout (reuse the quit_queue from main monitoring)
                        try:
                            quit_signal = quit_queue.get(timeout=check_interval)
                            if quit_signal == 'quit':
                                quit_requested = True
                                break
                        except queue.Empty:
                            pass
                
                # Handle quit if requested during retry monitoring
                if quit_requested:
                    # Force a final sync before showing confirmation to get accurate counts
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    print("Syncing task statuses before quit...")
                    sync_task_status()
                    
                    print(f"\n{'='*60}")
                    print("QUIT REQUESTED DURING RETRY MONITORING")
                    print(f"{'='*60}\n")
                    
                    n_running = len(launched_tasks) - len(completed_results)
                    n_completed = len([r for r in completed_results if r['status'] == 'completed' and objective_metric in r['metrics']])
                    
                    if n_completed == 0:
                        print(f"Quitting HPO - this will stop {n_running} still ongoing tasks.")
                        print("No completed tasks with valid metrics found - HPO will shut down without compiling results.")
                    else:
                        print(f"Quitting HPO - this will stop {n_running} still ongoing tasks and compile the HPO with {n_completed} completed tasks.")
                    
                    print("\nType 'agree' to proceed: ", end='', flush=True)
                    try:
                        confirmation = input().strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        confirmation = ''
                    
                    if confirmation == 'agree':
                        print(f"\n{'='*60}")
                        print("Shutting down HPO...")
                        print(f"{'='*60}\n")
                        
                        # Stop all running ClearML tasks and terminate all RunPod pods
                        from clearml import Task as TaskQuery
                        stopped_task_count = 0
                        terminated_pod_count = 0
                        
                        for task_info in launched_tasks:
                            # Skip tasks we've already collected
                            if any(r['task_name'] == task_info['task_name'] for r in completed_results):
                                continue
                            
                            # Try to stop ClearML task if it exists
                            clearml_task_stopped = False
                            try:
                                # Find the task
                                tasks = TaskQuery.get_tasks(
                                    project_name=project_name,
                                    task_name=task_info['task_name'],
                                    task_filter={'status': ['created', 'in_progress', 'queued']}
                                )
                                
                                if tasks:
                                    task = tasks[0] if len(tasks) == 1 else max(tasks, key=lambda t: t.data.created)
                                    task_created_time = task.data.created.timestamp()
                                    
                                    # Verify this is the task we launched
                                    if task_created_time >= task_info['launch_time']:
                                        print(f"Stopping ClearML task: {task_info['task_name']}")
                                        task.mark_stopped()
                                        stopped_task_count += 1
                                        clearml_task_stopped = True
                            except Exception as e:
                                print(f"Warning: Could not stop ClearML task {task_info['task_name']}: {e}")
                            
                            # Always try to terminate the RunPod pod, even if ClearML task doesn't exist yet
                            try:
                                pod_id = task_info['pod_id']
                                if not clearml_task_stopped:
                                    print(f"Terminating pod for {task_info['task_name']}: {pod_id}")
                                else:
                                    print(f"  Terminating pod: {pod_id}")
                                
                                launcher.terminate_pod(pod_id)
                                terminated_pod_count += 1
                            except Exception as e:
                                print(f"  Warning: Could not terminate pod {pod_id}: {e}")
                        
                        print(f"\nStopped {stopped_task_count} ClearML tasks and terminated {terminated_pod_count} RunPod pods.")
                        
                        if n_completed == 0:
                            # No completed tasks - just shut down orchestrator and exit
                            print("\nNo completed tasks to compile. Shutting down orchestrator.")
                            orchestrator_task.mark_stopped()
                            orchestrator_task.get_logger().report_text(
                                "HPO manually stopped with no completed tasks. No results compiled.",
                                print_console=False
                            )
                            print(f"\n{'='*60}")
                            print("HPO shut down.")
                            print(f"{'='*60}")
                            sys.exit(0)
                        
                        # Continue to results compilation with completed tasks
                        print(f"\nProceeding to compile results from {n_completed} completed tasks...\n")
                    else:
                        print("\nQuit cancelled, but retry monitoring has finished. Proceeding to results compilation...\n")
        
    # Generate final report (common to both normal and recovery modes)
    print(f"\n{'='*60}")
    if args.recovery or (not args.recovery and 'quit_requested' in locals() and quit_requested):
        print("Generating final report from completed tasks...")
    else:
        print("All tasks completed! Generating final report...")
    print(f"{'='*60}\n")
    
    # Sort by objective metric
    successful_results = [r for r in completed_results if r['status'] == 'completed' and objective_metric in r['metrics']]
    cuda_failed_results = [r for r in completed_results if r.get('exit_code') == EXIT_CUDA_UNAVAILABLE]
    retry_scheduled_results = [r for r in completed_results if r['status'] == 'retry_scheduled']
    
    if successful_results:
        # Determine if we're maximizing or minimizing
        maximize = objective_metric in ['avg_confidence', 'within_3px', 'within_5px', 'within_10px']
        successful_results.sort(key=lambda x: x['metrics'][objective_metric], reverse=maximize)
        
        best_result = successful_results[0]
        worst_result = successful_results[-1]
        
        # Create a results table for easy comparison
        import pandas as pd
        
        # Build comparison table
        table_data = []
        for i, result in enumerate(successful_results, 1):
            row = {
                'Rank': i,
                'Task ID': result['task_id'],
                objective_metric: f"{result['metrics'][objective_metric]:.4f}",
                'lr': result['params']['lr'],
                'batch_size': result['params']['batch_size'],
                'weight_decay': result['params']['weight_decay'],
                'scheduler': result['params'].get('scheduler_type', 'N/A'),
                'activation': result['params'].get('activation', 'N/A'),
                'gaussian_sigma': result['params'].get('gaussian_sigma', 'N/A'),
            }
            # Add other metrics if available
            if 'validation/loss' in result['metrics']:
                row['val_loss'] = f"{result['metrics']['validation/loss']:.6f}"
            if 'avg_distance' in result['metrics']:
                row['avg_dist'] = f"{result['metrics']['avg_distance']:.2f}"
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Log table to ClearML
        orchestrator_task.get_logger().report_table(
            title="HPO Results Comparison",
            series="All Experiments",
            iteration=0,
            table_plot=df
        )
        
        # Create console-friendly report
        manually_stopped_note = ""
        if not args.recovery and 'quit_requested' in locals() and quit_requested:
            manually_stopped_note = " (Manually stopped early)"
        
        final_report = f"""
{'='*80}
🏆 HPO FINAL REPORT{manually_stopped_note}
{'='*80}

Objective: {'MAXIMIZE' if maximize else 'MINIMIZE'} {objective_metric}
Total Experiments: {len(launched_tasks)}
Successful: {len(successful_results)}
Failed/Stopped: {len(launched_tasks) - len(successful_results)}
CUDA Unavailable (Machine Issues): {len(cuda_failed_results)}
Retries Triggered: {len(retry_scheduled_results)}

{'='*80}
🥇 BEST CONFIGURATION (Rank #1)
{'='*80}
{objective_metric}: {best_result['metrics'][objective_metric]:.4f}

Hyperparameters:
  • Learning Rate:    {best_result['params']['lr']}
  • Batch Size:       {best_result['params']['batch_size']}
  • Weight Decay:     {best_result['params']['weight_decay']}
  • Scheduler:        {best_result['params'].get('scheduler_type', 'N/A')}
  • Activation:       {best_result['params'].get('activation', 'N/A')}
  • CBAM:             {best_result['params'].get('cbam', 'N/A')}
  • Gaussian Sigma:   {best_result['params'].get('gaussian_sigma', 'N/A')}

Loss Weights:
  • pos_weight:       {best_result['params'].get('pos_weight', 'N/A')}
  • centroid_weight:  {best_result['params'].get('centroid_weight', 'N/A')}
  • sparsity_weight:  {best_result['params'].get('sparsity_weight', 'N/A')}
  • dice_weight:      {best_result['params'].get('dice_weight', 'N/A')}

Performance Metrics:"""
        
        # Separate loss components from other metrics
        loss_metrics = {}
        perf_metrics = {}
        for metric, value in best_result['metrics'].items():
            if metric.startswith('loss_') or metric == 'validation/loss':
                loss_metrics[metric] = value
            else:
                perf_metrics[metric] = value
        
        # Display performance metrics
        for metric, value in sorted(perf_metrics.items()):
            if isinstance(value, float):
                final_report += f"\n  • {metric:20s}: {value:.6f}"
            else:
                final_report += f"\n  • {metric:20s}: {value}"
        
        # Display loss components if available
        if loss_metrics:
            final_report += f"\n\nLoss Components (Weighted):"
            for metric, value in sorted(loss_metrics.items()):
                clean_name = metric.replace('loss_', '').replace('final_', '')
                if isinstance(value, float):
                    final_report += f"\n  • {clean_name:30s}: {value:.6f}"
                else:
                    final_report += f"\n  • {clean_name:30s}: {value}"
        
        final_report += f"\n\nClearML Task: https://app.clear.ml/projects/{project_name.replace(' ', '%20')}/experiments/{best_result['task_id']}"
        
        # Show improvement vs worst
        if len(successful_results) > 1:
            best_val = best_result['metrics'][objective_metric]
            worst_val = worst_result['metrics'][objective_metric]
            if maximize:
                improvement = ((best_val - worst_val) / worst_val * 100) if worst_val > 0 else 0
                final_report += f"\n\n📊 Improvement: {improvement:+.1f}% better than worst configuration"
            else:
                improvement = ((worst_val - best_val) / worst_val * 100) if worst_val > 0 else 0
                final_report += f"\n\n📊 Improvement: {improvement:.1f}% better than worst configuration"
        
        final_report += f"\n\n{'='*80}\n📋 RANKED RESULTS (Top {min(len(successful_results), 10)})\n{'='*80}\n"
        
        for i, result in enumerate(successful_results[:10], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            final_report += f"\n{medal} {objective_metric}: {result['metrics'][objective_metric]:.4f}"
            final_report += f"  |  lr={result['params']['lr']}, bs={result['params']['batch_size']}, wd={result['params']['weight_decay']}, {result['params'].get('scheduler_type', 'N/A')}, act={result['params'].get('activation', 'N/A')}, gs={result['params'].get('gaussian_sigma', 'N/A')}"
            final_report += f"\n   └─ Task: {result['task_id']}"
            if 'validation/loss' in result['metrics']:
                final_report += f"  (val_loss: {result['metrics']['validation/loss']:.6f})"
            final_report += "\n"
        
        final_report += f"\n{'='*80}\n"
        
        print(final_report)
        orchestrator_task.get_logger().report_text(final_report, print_console=False)
        
        # Also create a comparison plot
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Plot objective metric comparison
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'HPO Results Analysis - {objective_metric}', fontsize=16, fontweight='bold')
            
            # 1. Bar chart of all results
            ax = axes[0, 0]
            ranks = [r['Rank'] for r in table_data]
            values = [float(r[objective_metric]) for r in table_data]
            colors = ['gold' if i == 0 else 'silver' if i == 1 else 'chocolate' if i == 2 else 'steelblue' 
                     for i in range(len(ranks))]
            ax.bar(ranks, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Rank', fontweight='bold')
            ax.set_ylabel(objective_metric, fontweight='bold')
            ax.set_title(f'{objective_metric} by Rank')
            ax.grid(axis='y', alpha=0.3)
            
            # 2. Learning rate impact
            ax = axes[0, 1]
            lrs = [r['lr'] for r in table_data]
            ax.scatter(lrs, values, s=100, alpha=0.6, c=values, cmap='viridis', edgecolors='black')
            ax.set_xlabel('Learning Rate', fontweight='bold')
            ax.set_ylabel(objective_metric, fontweight='bold')
            ax.set_title('Learning Rate Impact')
            ax.set_xscale('log')
            ax.grid(alpha=0.3)
            
            # 3. Batch size impact
            ax = axes[1, 0]
            batch_sizes = [r['batch_size'] for r in table_data]
            unique_bs = sorted(set(batch_sizes))
            bs_avg = [np.mean([values[i] for i, bs in enumerate(batch_sizes) if bs == b]) for b in unique_bs]
            ax.bar(range(len(unique_bs)), bs_avg, color='coral', alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(unique_bs)))
            ax.set_xticklabels(unique_bs)
            ax.set_xlabel('Batch Size', fontweight='bold')
            ax.set_ylabel(f'Average {objective_metric}', fontweight='bold')
            ax.set_title('Batch Size Impact (Averaged)')
            ax.grid(axis='y', alpha=0.3)
            
            # 4. Scheduler comparison
            ax = axes[1, 1]
            schedulers = [r['scheduler'] for r in table_data]
            unique_sched = sorted(set(schedulers))
            sched_avg = [np.mean([values[i] for i, s in enumerate(schedulers) if s == sch]) for sch in unique_sched]
            ax.bar(range(len(unique_sched)), sched_avg, color='lightgreen', alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(unique_sched)))
            ax.set_xticklabels(unique_sched, rotation=15, ha='right')
            ax.set_xlabel('Scheduler Type', fontweight='bold')
            ax.set_ylabel(f'Average {objective_metric}', fontweight='bold')
            ax.set_title('Scheduler Impact (Averaged)')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # Log plot to ClearML
            orchestrator_task.get_logger().report_matplotlib_figure(
                title="HPO Analysis",
                series="Summary Plots",
                figure=fig,
                iteration=0
            )
            plt.close(fig)
        except Exception as e:
            print(f"⚠ Could not generate plots: {e}")
        
    else:
        failure_report = f"""
{'='*60}
HPO FINAL REPORT - NO SUCCESSFUL EXPERIMENTS
{'='*60}

Total Experiments: {len(launched_tasks)}
All experiments either failed or did not report the objective metric: {objective_metric}
CUDA Unavailable (Machine Issues): {len(cuda_failed_results)}

Task Status Summary:
"""
        for result in completed_results:
            exit_code_str = f" (exit_code: {result.get('exit_code', 'N/A')})" if 'exit_code' in result else ""
            failure_report += f"\n- {result['task_name']}: {result['status']}{exit_code_str}"
        
        failure_report += f"\n{'='*60}\n"
        
        print(failure_report)
        orchestrator_task.get_logger().report_text(failure_report, print_console=False)
    
    print(f"\n{'='*60}")
    print("HPO Complete!")
    print(f"Full report available in ClearML: {orchestrator_task.get_output_log_web_page()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
