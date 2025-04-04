"""
Aegear: a computer vision toolkit for tracking and analyzing fish behavior in controlled aquaculture environments.

Aegear is a modular, research-focused software package developed to support behavioral analysis of fish
within tank-based experimental setups, where movement, interaction, and environment must be studied in
a quantifiable and reproducible manner. The system was initially developed to support video-based 
tracking of *Acipenser gueldenstaedtii* (Russian sturgeon) juveniles during the larviculture experiments
conducted by Fazekas et al. (2025), with the goal of evaluating locomotory activity under varying 
feeding strategies and enrichment conditions.

The name *Aegear* is a deliberate wordplay combining *Ægir* — the Norse god of the sea, symbolizing
depth, mystery, and marine knowledge — with *eye-gear*, evoking the concepts of observation, precision,
and measurement. As a system that bridges mythic symbolism with modern visual sensing, Aegear is designed
as an extensible and high-performance tool for aquaculture research, starting with a focus on sturgeon but
potentially adaptable to other species.

Core components of Aegear include:

1. **Specialized object detection model**:
   Aegear introduces a U-Net-style convolutional neural network with an EfficientNet B0 backbone, trained via
   transfer learning and adapted for detection of Russian sturgeon juveniles. The network outputs spatial
   heatmaps representing object confidence, from which the centroid — defined by the peak response — serves
   both as a positional proxy and a scalar measure of detection confidence.

2. **Robust, drift-free tracking pipeline**:
   The tracking system is built around the heatmap-based detector and designed to ensure precision and
   temporal coherence under variable visual conditions. Particular attention was given to robustness against
   heterogeneous background textures, such as experimental tanks with pebble-covered floors, exhibiting
   varying density and reflectance. Once initialized, tracking operates frame-to-frame around the last known
   position, yielding fast and reliable tracking with negligible drift.

3. **Initialization via motion segmentation and sliding-window detection**:
   Initial detection is guided by background subtraction using the KNN-based method introduced by Zivkovic and
   van der Heijden (2006), as implemented in OpenCV's `cv2.createBackgroundSubtractorKNN`. Candidate motion
   regions are scanned using a lightweight sliding-window strategy. Once a confident detection is achieved,
   the system transitions into single-object tracking mode, allowing real-time or faster-than-real-time
   performance on modern consumer GPUs (e.g., NVIDIA RTX 3090 Ti).

4. **Scene calibration and metric measurements**:
   Aegear supports both intrinsic and extrinsic camera calibration for experiments where metric ground-truth
   data is required. Intrinsic calibration is performed using Zhang's method (1999), based on planar
   checkerboard images, implemented via OpenCV. Extrinsic calibration is achieved via user-defined reference
   points with known spatial coordinates in metric units. Although developed retrospectively — due to the
   unavailability of fiducial markers or fixed camera geometry during the original video acquisition — this
   hybrid calibration pipeline supports consistent real-world scaling of trajectories and enables total
   distance measurements in centimeters.

---

**References**:

Fazekas, G., Müller, T., Berzi-Nagy, L., Ljubobratović, R., Stanivuk, J., Fazekas, D. L., Káldy, J., Vass, N., 
& Ljubobratović, U. (2025). *The feeding strategy and environmental enrichment modulate the locomotory 
activity in Russian sturgeon (Acipenser gueldenstaedtii) juveniles - pursuing an optimal factorial 
combination of larviculture strategies*. Research Center for Fisheries and Aquaculture (HAKI), 
Hungarian University of Agriculture and Life Sciences (MATE), Szarvas, Hungary.

Zivkovic, Z., & van der Heijden, F. (2006). *Efficient adaptive density estimation per image pixel for the task 
of background subtraction*. Pattern Recognition Letters, 27(7), 773-780.

Zhang, Z. (2000). *A flexible new technique for camera calibration*. IEEE Transactions on Pattern Analysis and 
Machine Intelligence, 22(11), 1330-1334.

Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
In *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, Lecture Notes in Computer Science, vol 9351.

Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*.
In *Proceedings of the 36th International Conference on Machine Learning (ICML)*.

"""
