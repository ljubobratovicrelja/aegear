# Aegear


**Tracking and analyzing fish behavior in controlled aquaculture environments**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)[![Documentation](https://img.shields.io/badge/documentation-online-brightgreen.svg?style=flat&logo=readthedocs)](https://ljubobratovicrelja.github.io/aegear/)[![Docker](https://img.shields.io/badge/docker-aegear-blue?logo=docker&logoColor=white)](https://hub.docker.com/r/ljubobratovicrelja/aegear)



<p align="center">
  <img src="docs/media/logo.png" alt="ÆGEAR Logo" width="35%"/>
</p>

---


## 🧠 Project Overview

**Aegear** is a computer vision toolkit developed for the analysis of fish locomotion in controlled aquaculture environments. Originally designed for behavioral studies on juvenile Russian sturgeon (*Acipenser gueldenstaedtii*), the system enables robust detection and tracking of individual fish across a range of experimental conditions, including tanks with textured floors and heterogeneous lighting.

The toolkit addresses the need for accurate, reproducible behavioral metrics in video-based aquaculture experiments. It provides a complete pipeline for fish localization, trajectory tracking, scene calibration, and data augmentation — with a focus on modularity, reusability, and extensibility to other species and experimental setups.

The name **Aegear** references **Ægir**, the Norse god of the sea, symbolizing the system's focus on aquatic environments, while also invoking *eye-gear* — a metaphor for visual instrumentation and observation.

---


<p align="center">
  <img src="docs/media/1.png" alt="Dense Tank" width="45%" style="margin-right:2%;"/>
  <img src="docs/media/2.png" alt="Open Arena" width="45%"/>
</p>

---

## 🔬 Project Summary

Aegear is a computer vision toolkit for detecting and tracking fish in aquaculture environments. Initially applied in the doctoral research of Georgina Fazekas (2020– ), which investigated how environmental and feeding strategies affect the swimming activity of juvenile sturgeons (Acipenser gueldenstaedtii, A. ruthenus), Aegear was developed to address the limitations of existing animal tracking tools in real-world aquaculture conditions. Unlike systems such as idtracker.ai (Romero-Ferrero et al., 2018), which require clean backgrounds and controlled lighting, Aegear is designed for noisy environments with textured floors, variable illumination, and water reflections.

At its core, Aegear integrates a U-Net-style detection network with an EfficientNet-B0 encoder backbone, specialized through transfer learning on custom tank recordings. For tracking, the same backbone extends into a Siamese network architecture that associates detections across frames, enabling appearance-based localization without manual re-identification. This combination ensures robust trajectory reconstruction even in complex scenes.

The toolkit also includes camera calibration routines for intrinsic parameter estimation and extrinsic metric scaling from four predefined scene reference points, allowing trajectories to be expressed in real-world metric units (e.g., centimeters).

---

## 🚧 Known Limitations

- Currently limited to **single-object tracking**; no support yet for multi-class or multi-fish tracking.
- The detection model is trained on sterlet (*Acipenser ruthenus*) and Russian sturgeon (*Acipenser gueldenstaedtii*) video data and likely requires additional training for species with significantly different shapes or swimming patterns.

---

## 📦 Installation
Aegear can be installed either for development use in notebooks or as a GUI-based application.

🔧 Development / Notebook Usage
To install Aegear in editable mode with additional dependencies for training, notebooks, and analysis:

```bash
git clone https://github.com/ljubobratovicrelja/aegear.git
cd aegear
pip install -e .[dev]
```

**Note**: Best do this in an virtual environment because of heavy dependencies that Aegear carries along with its install.

This mode is ideal for working with Jupyter notebooks or customizing the codebase.

---

## 🚀 GUI Application (Early Access)

Aegear includes a desktop GUI built with Tkinter. Once installed, the app can be launched via:

```bash
aegear-gui
```

This requires Python ≥3.10 and a working Tkinter environment. Therefore make sure that Tkinter and Python with Tk support is installed on your machine.

### 🛠️ Prebuilt binaries

There are also included binaries made using PyInstaller for Win64 and macOS machines with Apple Silicon within [the release packages](https://github.com/ljubobratovicrelja/aegear/releases). However keep in mind that Windows builds include the basic PyTorch version (without CUDA). To run `aegear-gui` with CUDA, remove the preinstalled torch and [install your pytorch version](https://pytorch.org/) matching your installed CUDA runtime (Aegear models are tested using CUDA Toolkit version 12.4).

> ⚠️ Experimental Notice:
> The current GUI is designed specifically for the video format and calibration workflow used in the original Russian sturgeon experiments. It assumes a specific directory structure and input format.
> A more flexible and general-purpose GUI for broader use cases is under active development.

---

## 🤝 Contributions & Collaboration

Aegear was originally developed around a single research project in controlled aquaculture environments. While it is currently tailored to tracking fish under specific conditions, we envision Aegear growing into a more general-purpose toolkit for animal tracking in both **academic** and **industrial** settings.

We warmly invite:  
- 🧑‍🔬 Researchers in **biology, ethology, aquaculture**, or other animal behavior fields  
- 🏭 Practitioners in **industrial monitoring of animal populations**  
to explore Aegear and contact us for support or potential collaboration.  

If your use case involves different species, environments, or tracking requirements, we are happy to:  
- Extend Aegear for broader animal tracking scenarios  
- Discuss customizations and new features  
- Work together on shared challenges in visual tracking systems  

> 📌 **Feature Requests:** Open a GitHub issue if you require specific capabilities not yet available. We will prioritize these to make Aegear a useful resource for the wider community.

---

## 🧠 Acknowledgments

Special thanks to Gina and Uroš from the HAKI research team, whose collaboration and encouragement sparked the development of this toolkit.

## 📖 References


> Fazekas, G.: Investigating the effects of environmental factors and feeding strategies on early life development and behavior of Russian sturgeon (Acipenser gueldenstaedtii) and sterlet (A. ruthenus) [Doctoral thesis].
> Hungarian University of Agriculture and Life Sciences (MATE), Hungary.  
>
> Romero-Ferrero, F., Bergomi, M. G., Hinz, R., Heras, F. J. H., & de Polavieja, G. G. (2018).
> idtracker.ai: tracking all individuals in small or large collectives of unmarked animals.
> Nature Methods, 16(2), 179–182. [arXiv:1803.04351]
>
> Tan, M., & Le, Q. V. (2019).
> EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
> Proceedings of the 36th International Conference on Machine Learning, PMLR 97:6105–6114. arXiv:1905.11946
>
> Bertinetto, L., Valmadre, J., Henriques, J. F., Vedaldi, A., & Torr, P. H. S. (2016).
> Fully-Convolutional Siamese Networks for Object Tracking.
> European Conference on Computer Vision (ECCV) Workshops. arXiv:1606.09549
