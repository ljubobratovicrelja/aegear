# Aegear

**Tracking and analyzing fish behavior in controlled aquaculture environments**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)  
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)  
[![Documentation](https://img.shields.io/badge/docs-link-blue.svg)](#)  
[![PyPI](https://img.shields.io/badge/pypi-coming_soon-orange.svg)](#)

---

## 🧠 Project Overview

**Aegear** is a computer vision toolkit for fish behavior analysis in aquaculture research settings. Originally developed to support behavioral experiments on juvenile Russian sturgeon (*Acipenser gueldenstaedtii*), Aegear enables drift-free, metric-accurate tracking of fish in complex tank environments using a deep learning-based pipeline.

It includes tools for camera calibration, model training, and data augmentation, with the goal of facilitating reproducible experiments and promoting adaptive reuse for other species in aquaculture.

The name is a play on **Ægir**, the Norse god of the sea, evoking depth, observation, and aquatic control — as well as **eye-gear**, a nod to vision-based analysis systems.

---

## 📦 Features

- 🔍 **U-Net + EfficientNet B0** detection model, transfer-learned and specialized for Russian sturgeon.
- 🔥 **Heatmap-based centroid tracking** using peak activation as a proxy for fish position and confidence.
- 🎯 **Sliding-window tracking initialization** using motion segmentation (OpenCV's KNN background subtractor).
- ⚡ **Real-time tracking** on CUDA GPUs (e.g., RTX 3090 Ti).
- 🧭 **Scene calibration** for metric distance measurement via intrinsic (Zhang) and extrinsic user-defined 4-point systems.
- 🧪 **Notebooks** for training, validation visualization, and dataset augmentation.
- 🛠️ **Helper scripts** for camera calibration and training ROI generation from video frames.
- 📁 **COCO-format dataset preparation**, with augmentation across background complexity and heatmap generation.
- ✅ MIT licensed.

---

## 📚 Publications & Citations

Aegear was developed for the larviculture experiments led by Fazekas et al. (2025). Citation details:

> Fazekas, G., Müller, T., Berzi-Nagy, L., Ljubobratović, R., Stanivuk, J., Fazekas, D. L., Káldy, J., Vass, N.,  
> & Ljubobratović, U. (2025). *The feeding strategy and environmental enrichment modulate the locomotory  
> activity in Russian sturgeon (Acipenser gueldenstaedtii) juveniles – pursuing an optimal factorial  
> combination of larviculture strategies*. Research Center for Fisheries and Aquaculture (HAKI),  
> Hungarian University of Agriculture and Life Sciences (MATE), Szarvas, Hungary.

Other core references:

- Zivkovic & van der Heijden (2006), background subtraction with adaptive KNN  
- Zhang (2000), flexible camera calibration  
- Ronneberger et al. (2015), U-Net architecture  
- Tan & Le (2019), EfficientNet backbone

(See full citations in `aegear/__init__.py`)

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
├── calibration/           # Scripts for intrinsic/extrinsic camera calibration
├── tools/                 # Training ROI labeling tools, background blending
├── notebooks/             # Jupyter notebooks for training, validation, COCO data prep
│   ├── training.ipynb
│   ├── coco_prep.ipynb
│
├── models/                # Saved weights and training checkpoints
├── data/                  # COCO-style datasets and training samples
│   ├── images/
│   └── annotations.json
```

---

## 🚧 Known Limitations

- Currently limited to **single-object tracking**; no support yet for multi-class or multi-fish tracking.
- The detection model is specialized for Russian sturgeon and must be retrained for other species.

---

## 📈 Future Work

- [ ] Extend detection to multi-species, multi-object support  
- [ ] Add fish ID tracking with re-identification (Re-ID)  
- [ ] Export tracking results to standardized formats (CSV, JSON)  
- [ ] Enable live stream tracking from connected cameras  

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 📦 Installation (Coming Soon)

Aegear will be published on PyPI. Until then:

```
git clone https://github.com/your-username/aegear.git  
cd aegear  
pip install -r requirements.txt
```

---

## 🧠 Acknowledgments

Special thanks to the HAKI research team for their trust and scientific collaboration, and to the dedicated aquaculture biologists whose expertise shaped the problem domain.
