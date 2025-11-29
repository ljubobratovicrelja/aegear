
# 🐳 Docker Support

Aegear provides a ready-to-use Docker image to simplify deployment and training in cloud environments or on systems without a configured Python environment.

---

## 📦 Public Docker Image

The official Aegear Docker image is hosted on [Docker Hub]():

```
ljubobratovicrelja/aegear:latest
```

Pull the image with:

```bash
docker pull ljubobratovicrelja/aegear:latest
```

---

## 🚀 Running the Image

Launch a container with GPU support (if available):

```bash
docker run --gpus all -it ljubobratovicrelja/aegear:latest
```

To mount a local directory for accessing datasets or saving models:

```bash
docker run --gpus all -it -v /path/to/data:/workspace/data ljubobratovicrelja/aegear:latest
```

This will mount `/path/to/data` on your host to `/workspace/data` inside the container.

---

## 🛠️ Training with Docker

The recommended workflow for training models is to use the CLI-based training script via the container entrypoint. The image includes all dependencies and uses `run_training.sh` to configure and launch training.

**Example:**

```bash
docker run --gpus all -e MODEL_TYPE=efficient_unet -e DATA_MANIFEST=/workspace/data/manifest.json -e MODEL_DIR=/workspace/models/unet -e CHECKPOINT_DIR=/workspace/models/unet/checkpoints ljubobratovicrelja/aegear:latest
```

Set additional environment variables to customize training (see [docker/README.md](https://github.com/ljubobratovicrelja/aegear/blob/hpo/docker/README.md) for details).

---

## ⚠️ Notebook-based Training (Deprecated)

Notebook-based training (`training_unet.ipynb`, `training_siamese.ipynb`) is still available for development, but is deprecated and will be removed in future releases. Please use the CLI training workflow for new projects.

