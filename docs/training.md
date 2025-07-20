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

## 📖 Notes

- Training requires PyTorch with GPU support for reasonable performance.
- Pretrained EfficientNet-B0 weights are loaded automatically.
- CBAM modules enhance spatial and channel attention, improving performance in cluttered environments.
- Data augmentations are applied at runtime to improve generalization.
- The Siamese tracker is highly sensitive to template drift; careful dataset preparation is recommended.

---

## 📜 References

See the original papers for the underlying architectures:  
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. [arXiv:1807.06521](https://arxiv.org/abs/1807.06521)  
- Tan, M., & Le, Q. V. (2019). EfficientNet. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)  
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)  
- Bertinetto, L., et al. (2016). Fully-Convolutional Siamese Networks. [arXiv:1606.09549](https://arxiv.org/abs/1606.09549)