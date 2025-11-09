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

The datasets are defined in `aegear.datasets` and support flexible configurations:  

- **FishHeatmapDataset**
  - Used for U-Net training.
  - Inputs: Grayscale or RGB video frames.
  - Labels: Gaussian-blurred fish centroids as heatmaps.

- **SiameseFishDataset**
  - Used for Siamese tracker training.
  - Pairs template and search images with offset targets.

**Data Augmentations**:

  - Random cropping and scaling.
  - Flips and rotations.
  - Brightness and contrast adjustments.

**📥 Public Dataset**:

A preprocessed dataset for our training is publicly available:

  - [Detection (U-Net) Dataset](https://storage.googleapis.com/aegear-training-data/cache/detection.zip)

  - [Tracking (Siamese) Dataset](https://storage.googleapis.com/aegear-training-data/cache/tracking.zip)

---

## 🔥 Training Scripts

Training scripts are located in the `notebooks/` directory:  

| Notebook                  | Purpose                 |
|---------------------------|-------------------------|
| `training_unet.ipynb`     | Train EfficientUNet     |
| `training_siamese.ipynb`  | Train Siamese Tracker   |

Each notebook provides step-by-step setup for loading datasets, initializing models, defining optimizers, and launching training loops. They are Jupyter notebooks to allow easy modification for custom datasets.

---

## 🚀 Training on Custom Dataset

To train Aegear on your own data:  

1. **Prepare Dataset**  
   - Structure your dataset similarly to the existing `FishHeatmapDataset` and `SiameseFishDataset` expectations.
   - For U-Net:
     - Frames and corresponding centroid annotations.
   - For Siamese Tracker:
     - Frame pairs with ground-truth offsets.

2. **Edit Configuration**  
   - Update dataset paths and parameters in the training notebooks (`training_unet.ipynb`, `training_siamese.ipynb`).

3. **Run Training**  
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook notebooks/training_unet.ipynb
     jupyter notebook notebooks/training_siamese.ipynb
     ```

4. **Export Models**  
   - Save trained weights in `data/models/` for Aegear to use in the GUI.

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