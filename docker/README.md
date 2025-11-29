# Aegear Docker Image & Training Entrypoint

This directory contains the Dockerfile and entrypoint script for building and running the Aegear training environment in a containerized, reproducible way. The image is designed for both cloud and local training workflows, and can be used independently of RunPod or any specific cloud provider.

## Docker Image Overview

- **Base Image:** `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime` (includes CUDA 12.8, cuDNN 9, and PyTorch)
- **Entrypoint Script:** `run_training.sh` (copied into `/app/run_training.sh`)
- **Branch Selection:** The image supports runtime selection of the Aegear git branch via the `AEGEAR_BRANCH` environment variable (default: `main`).
- **System Dependencies:** Installs build tools, git, ffmpeg, and other required packages for training and data processing.

## Building the Image

To build the Docker image locally:

```bash
docker build -t aegear-train:latest .
```

## Entrypoint: `run_training.sh`

This script is the main entrypoint for training. It:
- Clones the Aegear repository into a unique directory inside the container
- Installs Aegear and its training dependencies
- Optionally installs ClearML for experiment tracking (if credentials are provided)
- Validates CUDA/GPU availability
- Assembles training arguments from environment variables
- Runs the training script (`tools/train.py`) with the specified configuration
- Handles output directories and checkpointing
- Prevents restart loops in cloud environments by using sentinel files

### Key Environment Variables

| Variable                | Description                                                      | Example Value                  |
|------------------------ |------------------------------------------------------------------|-------------------------------|
| `AEGEAR_BRANCH`         | Git branch to clone (default: `main`)                            | `hpo`                         |
| `MODEL_TYPE`            | Model type to train (`efficient_unet` or `siamese`)              | `efficient_unet`              |
| `DATA_MANIFEST`         | Path to dataset manifest JSON                                    | `/workspace/data/manifest.json`|
| `MODEL_DIR`             | Directory for model outputs                                      | `/workspace/models/unet`       |
| `CHECKPOINT_DIR`        | Directory for checkpoints                                        | `/workspace/models/unet/checkpoints`|
| `BATCH_SIZE`            | Training batch size                                              | `128`                         |
| `EPOCHS`                | Number of training epochs                                        | `10`                          |
| `DEVICE`                | Device to use (`cpu`, `cuda`, `auto`)                            | `cuda`                        |
| `CLEARML_TASK`          | (Optional) ClearML task name for experiment tracking             | `unet_exp_001`                |
| `CLEARML_PROJECT`       | (Optional) ClearML project name                                  | `aegear`                      |
| ...                     | Many more options are supported (see `run_training.sh` and `train.py`)

All arguments can be set as environment variables when launching the container. The script will assemble the correct CLI arguments for the training script.

### Example: Local Training

```bash
docker run --gpus all \
  -e MODEL_TYPE=efficient_unet \
  -e DATA_MANIFEST=/workspace/data/manifest.json \
  -e MODEL_DIR=/workspace/models/unet \
  -e CHECKPOINT_DIR=/workspace/models/unet/checkpoints \
  -e BATCH_SIZE=64 \
  -e EPOCHS=20 \
  -v /local/data:/workspace/data \
  -v /local/models:/workspace/models \
  aegear-train:latest
```

You can set any supported environment variable to customize the training run. Output and checkpoints will be written to the mounted `/workspace` volume.

## Usage in Cloud Training

This image and entrypoint are used as the foundation for cloud training workflows, including:
- **RunPod:** Automated pod launching and training orchestration via `launch_runpod_training.py` and HPO via `clearml_runpod_hpo.py`.
- **Other Clouds:** The image can be used in any environment that supports Docker and GPU passthrough (AWS, GCP, Azure, on-prem clusters, etc.).

## Standalone Usage

You do not need RunPod or ClearML to use this image. Simply set the required environment variables and mount your data/model directories as needed. The training script will run as specified and output results to `/workspace`.


---

### Prebuilt Image on Docker Hub

If you prefer not to build the image locally, a prebuilt version is available on Docker Hub:

```
docker pull ljubobratovicrelja/aegear:latest
```

You can use this image directly in any compatible environment (local, cloud, or services like RunPod) by replacing `aegear-train:latest` with `ljubobratovicrelja/aegear:latest` in your `docker run` commands.

---

For questions or issues, feel free to contact the maintainer or open an issue.
