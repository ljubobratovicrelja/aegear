import os
import argparse
from typing import Dict

from .runpod_launcher import RunPodLauncher, PodManager

__all__ = ['RunPodLauncher', 'PodManager', 'build_training_env_vars']


def build_training_env_vars(args: argparse.Namespace) -> Dict[str, str]:
    env = {
        "TASK_NAME": args.task_name,
        "AEGEAR_BRANCH": args.branch,
        "MODEL_TYPE": args.model_type,
        "DATA_MANIFEST": args.data_manifest,
        "MODEL_DIR": args.model_dir,
        "CHECKPOINT_DIR": args.checkpoint_dir,
        "DEVICE": args.device if args.device else "cuda",
    }
    if args.batch_size:
        env["BATCH_SIZE"] = str(args.batch_size)
    if args.train_ratio:
        env["TRAIN_RATIO"] = str(args.train_ratio)
    if args.num_workers is not None:
        env["NUM_WORKERS"] = str(args.num_workers)
    if args.gaussian_sigma:
        env["GAUSSIAN_SIGMA"] = str(args.gaussian_sigma)
    if args.weights:
        env["WEIGHTS"] = args.weights
    if args.pretrained_model_dir:
        env["PRETRAINED_MODEL_DIR"] = args.pretrained_model_dir
    if args.epochs:
        env["EPOCHS"] = str(args.epochs)
    if args.lr:
        env["LR"] = str(args.lr)
    if args.epoch_vis:
        env["EPOCH_VIS"] = args.epoch_vis
    if args.epoch_save_interval:
        env["EPOCH_SAVE_INTERVAL"] = str(args.epoch_save_interval)
    if args.weight_decay:
        env["WEIGHT_DECAY"] = str(args.weight_decay)
    if args.activation:
        env["ACTIVATION"] = args.activation
    if args.seed:
        env["SEED"] = str(args.seed)
    if args.continue_training:
        env["CONTINUE_TRAINING"] = "1"
    if args.use_best_model:
        env["USE_BEST_MODEL"] = "1"
    if args.cbam:
        env["CBAM"] = "1"
    if args.use_visualizer:
        env["USE_VISUALIZER"] = "1"
    if args.autodownload:
        env["AUTODOWNLOAD"] = "1"
    if args.verbose:
        env["VERBOSE"] = "1"
    if args.training_stages:
        env["TRAINING_STAGES"] = args.training_stages
    if args.loss_params:
        env["LOSS_PARAMS"] = args.loss_params
    if args.scheduler_type:
        env["SCHEDULER_TYPE"] = args.scheduler_type
    if args.scheduler_params:
        env["SCHEDULER_PARAMS"] = args.scheduler_params
    if args.clearml_task:
        env["CLEARML_TASK"] = args.clearml_task
    if args.clearml_project:
        env["CLEARML_PROJECT"] = args.clearml_project
    clearml_key = os.getenv("CLEARML_API_ACCESS_KEY")
    clearml_secret = os.getenv("CLEARML_API_SECRET_KEY")
    if clearml_key:
        env["CLEARML_API_ACCESS_KEY"] = clearml_key
    if clearml_secret:
        env["CLEARML_API_SECRET_KEY"] = clearml_secret
    env["CLEARML_API_SERVER"] = "https://api.clear.ml"
    env["CLEARML_WEB_SERVER"] = "https://app.clear.ml"
    env["CLEARML_FILES_SERVER"] = "https://files.clear.ml"

    return env