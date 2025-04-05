# Aegear

**Tracking and analyzing fish behavior in controlled aquaculture environments**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)[![Documentation](https://img.shields.io/badge/docs-link-blue.svg)](#)[![PyPI](https://img.shields.io/badge/pypi-coming_soon-orange.svg)](#)

<p align="center">
  <img src="media/logo.png" alt="ÆGEAR Logo" width="500"/>
</p>

---

## 🧠 Project Overview

**Aegear** is a computer vision toolkit developed for the analysis of fish locomotion in controlled aquaculture environments. Originally designed for behavioral studies on juvenile Russian sturgeon (*Acipenser gueldenstaedtii*), the system enables robust detection and tracking of individual fish across a range of experimental conditions, including tanks with textured floors and heterogeneous lighting.

The toolkit addresses the need for accurate, reproducible behavioral metrics in video-based aquaculture experiments. It provides a complete pipeline for fish localization, trajectory tracking, scene calibration, and data augmentation — with a focus on modularity, reusability, and extensibility to other species and experimental setups.

The name **Aegear** references **Ægir**, the Norse god of the sea, symbolizing the system's focus on aquatic environments, while also invoking *eye-gear* — a metaphor for visual instrumentation and observation.

---

## 🔬 Project Summary

At the core of Aegear is a deep learning model for spatial fish localization, built on a U-Net-style architecture with an EfficientNet B0 encoder backbone. The encoder is initialized from `torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')`, with the first three stages frozen during training to preserve generic visual features. Decoder layers perform progressive upsampling using transposed convolutions, with skip connections linking each encoder stage to its corresponding decoder layer.

The model produces a single-channel heatmap that reflects the likelihood and position of the fish within the input frame. Supervision is carried out using a weighted binary cross-entropy loss that emphasizes central activations, combined with a custom centroid distance loss to directly penalize spatial errors in predicted heatmap peaks. This training objective ensures precise localization under class imbalance and subtle visual cues.

Tracking is initialized using a sliding-window search constrained by motion segmentation via OpenCV’s KNN background subtraction algorithm (Zivkovic & van der Heijden, 2006). Once the target is detected, subsequent frames are processed with local search around the last known position. This localized tracking strategy offers robust performance in dynamic or noisy visual conditions and supports real-time execution on modern CUDA-enabled GPUs.

To allow for real-world quantification, Aegear includes a calibration module for metric scaling. Intrinsic parameters are obtained using Zhang’s method based on checkerboard imagery, while extrinsic calibration is performed by selecting four known reference points within the tank environment. This enables accurate reconstruction of fish trajectories in metric units (e.g., centimeters), suitable for downstream analysis of activity levels and behavioral patterns.

In addition to the main pipeline, Aegear includes tools for:
- camera calibration and manual ROI annotation from videos,
- COCO-style dataset generation and polygon-to-heatmap conversion using distance transforms,
- synthetic dataset augmentation by compositing fish onto complex backgrounds.

---


## 📚 Publications & Citations

Aegear was originally developed for the larviculture experiments led by Fazekas et al. (2025). Citation details:

> Fazekas, G., Müller, T., Berzi-Nagy, L., Ljubobratović, R., Stanivuk, J., Fazekas, D. L., Káldy, J., Vass, N.,  
> & Ljubobratović, U. (2025). *The feeding strategy and environmental enrichment modulate the locomotory  
> activity in Russian sturgeon (Acipenser gueldenstaedtii) juveniles – pursuing an optimal factorial  
> combination of larviculture strategies*. Research Center for Fisheries and Aquaculture (HAKI),  
> Hungarian University of Agriculture and Life Sciences (MATE), Szarvas, Hungary.

However, in the meantime it has evolved quite so much in its methods and results that are achieved, but the objective
itself is unchanged.

---

## 🔧 Requirements

Python 3.10+

```
matplotlib==3.7.2
moviepy==2.1.2
numpy==2.2.4
opencv_contrib_python==4.11.0.86
opencv_python==4.11.0.86
Pillow==11.1.0
pyinstaller==6.12.0
scipy==1.15.2
torch==2.6.0+cu124
torchvision==0.21.0+cu124
```

---

## 🗂 Repository Structure

```
aegear/
│   __init__.py            # Main module with detailed project description
│
├── tools/                 # Training ROI labeling tools, camera calibration
├── notebooks/             # Jupyter notebooks for training, validation, COCO data prep
│   ├── training.ipynb
│   ├── coco_prep.ipynb
│
├── data/                  # Various data points required for the project.
|   ├── models/            # Saved weights and training checkpoints
```

---

## 🚧 Known Limitations

- Currently limited to **single-object tracking**; no support yet for multi-class or multi-fish tracking.
- The detection model is specialized for Russian sturgeon and must be retrained for other species.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 📦 Installation (Coming Soon)

Aegear will be published on PyPI. Until then:

```
git clone https://github.com/ljubobratovicrelja/aegear.git  
cd aegear  
pip install -r requirements.txt
```

---

## 🧠 Acknowledgments

Special thanks Gina and Uros from the HAKI research team for their trust and scientific collaboration on this exciting project!
