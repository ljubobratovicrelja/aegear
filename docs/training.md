# 🏋️‍♂️ Network Training Overview

This page provides an overview of the training process for Aegear's neural networks, including details on datasets, model architectures, and a guide for training on custom datasets.

---

## 📦 Networks Overview

Aegear uses two deep learning models:  

### 1️⃣ EfficientUNet (Detection)
- **Architecture**: U-Net-style with an **EfficientNet-B0 encoder** and integrated **Convolutional Block Attention Module (CBAM)** for channel and spatial attention refinement.
- **Encoder**: Pretrained on ImageNet, early layers frozen during training to preserve general visual features.
- **Decoder**: Transposed convolutions with skip connections for spatial resolution recovery.
- **Output**: Single-channel heatmap indicating fish presence and location.
- **Loss Function**:
  - Weighted binary cross-entropy for class imbalance.
  - Centroid distance penalty to improve spatial accuracy.

### 2️⃣ Siamese Tracker (Tracking)
- **Architecture**: Siamese network sharing the same EfficientNet-B0 + CBAM backbone.
- **Inputs**: Template (last ROI) and search region (current ROI).
- **Output**: High-resolution response heatmap for localization.
- **Loss Function**:
  - Response map peak loss (cross-entropy and L2 penalty).

---

## 🗂️ Dataset Setup

Aegear uses a unified dataset format for both detection and tracking model training: `WebTrackingDatasetWithLength`.

- Data is stored in tar shard files, described by a manifest JSON.
- The loader function `load_dataset_from_shards` (see [`aegear.nn.datasets`](https://github.com/ljubobratovicrelja/aegear/blob/main/src/aegear/nn/datasets.py)) automatically splits shards into training and validation sets, and builds DataLoaders for both workflows.
- The same dataset structure is used for both EfficientUNet and Siamese models.

**📥 Public Data Bucket:**

Training data shards are available for download from our public Google Cloud Storage bucket:

```
gs://aegear-training-data/shards
```

---

## 🔥 Training Workflow

### Recommended: CLI Training Script

For new projects and production training, use the CLI-based training script:

- `tools/train.py` — Flexible command-line training for EfficientUNet and Siamese models.
- Supports all major training options via CLI arguments and environment variables.
- Designed for use with the Aegear Docker image (see [docker.md](docker.md) for container usage and cloud deployment).

**Example:**

```bash
python tools/train.py --model-type efficient_unet --data-manifest /path/to/manifest.json --model-dir /path/to/models --checkpoint-dir /path/to/checkpoints --epochs 10 --batch-size 128
```

For containerized and cloud training, see [docker.md](docker.md).

For cloud launching, hyperparameter optimization (HPO), and experiment orchestration, see [Cloud Training & HPO Pipeline](cloud_training.md).

---

### ⚠️ Notebook-based Training (Deprecated)

The legacy training notebooks (`notebooks/training_unet.ipynb`, `notebooks/training_siamese.ipynb`) are still available for development and experimentation, but are deprecated and will be removed in future releases. Please migrate to the CLI training workflow for all new work.

---

## 🔍 Model Evaluation and Dataset Inspection

After training, you can evaluate your models and inspect predictions using the dataset inspection tool. The project includes a CLI utility provides visual analysis through FiftyOne, allowing you to review model predictions, identify failure cases, and measure performance metrics.

### Prerequisites

The inspection tool requires FiftyOne, which is included in the `dev` optional dependencies. If you haven't already installed Aegear with development tools:

```bash
pip install -e .[dev]
```

### Basic Usage

Evaluate a tracking model on the validation set:

```bash
python tools/dataset_inspection.py tracking --dataset-name 4_per_23
```

Evaluate a detection model:

```bash
python tools/dataset_inspection.py detection --dataset-name my_detection_dataset
```

### Command-Line Options

The inspection tool supports the following arguments:

| Argument | Description |
|----------|-------------|
| `mode` | Model type: `tracking` or `detection` |
| `--dataset-name` | Name of cached dataset to evaluate |
| `--custom-path` | Path to custom dataset (alternative to `--dataset-name`) |
| `--model-path` | Specific model checkpoint to use (auto-detects latest if omitted) |
| `--models-dir` | Directory containing model checkpoints (default: `models/`) |
| `--batch-size` | Inference batch size (default: 128) |
| `--num-workers` | Data loading workers (default: 4) |
| `--device` | Device: `cuda`, `cpu`, or `auto` (default: `auto`) |
| `--fiftyone-name` | Custom name for FiftyOne dataset |
| `--no-launch` | Build dataset without launching viewer |
| `--skip-download` | Skip automatic dataset download |

### Example Workflows

**Evaluate with Custom Model:**

```bash
python tools/dataset_inspection.py tracking \
    --dataset-name 4_per_23 \
    --model-path models/model_siamese_2025-01-15.pth
```

**Use Custom Dataset:**

```bash
python tools/dataset_inspection.py tracking \
    --custom-path /path/to/my/validation/data
```

**Batch Processing (No Viewer Launch):**

```bash
python tools/dataset_inspection.py tracking \
    --dataset-name 4_per_23 \
    --fiftyone-name eval-experiment-1 \
    --no-launch

# Launch viewer separately when ready
fiftyone app launch eval-experiment-1
```

### Visualization Features

The FiftyOne viewer provides:

**For Tracking Models:**

- Predicted and ground truth heatmaps
- Predicted and ground truth keypoints with confidence scores
- Template and search image paths
- Template/search ROI bounding boxes (when available)
- Per-sample distance error metrics
- Background sample tagging

**For Detection Models:**

- Predicted and ground truth heatmaps
- Predicted and ground truth keypoints with confidence scores
- Per-sample distance error metrics
- Background sample tagging

### Performance Metrics

The tool automatically computes:

- Euclidean distance between predicted and ground truth centroids
- Model confidence scores per prediction
- Success/failure rates across the validation set

Use FiftyOne's filtering capabilities to analyze specific subsets, such as high-error samples or low-confidence predictions.

---

## 📜 References

See the original papers for the underlying architectures:
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. [arXiv:1807.06521](https://arxiv.org/abs/1807.06521)
- Tan, M., & Le, Q. V. (2019). EfficientNet. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Bertinetto, L., et al. (2016). Fully-Convolutional Siamese Networks. [arXiv:1606.09549](https://arxiv.org/abs/1606.09549)