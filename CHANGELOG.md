# Release Notes


## v0.4.0

-   **Model Updates**
    -   **Enhanced Model Architecture**: Integrated **CBAM (Convolutional Block Attention Module)** into both the detection (EfficientUNet) and tracking (SiameseTracker) models. This enhances feature representation by applying channel and spatial attention, improving the models' ability to focus on relevant information.
    -   **Optimized EfficientUNet**: The `EfficientUNet` architecture has been streamlined by removing deeper layers and introducing a new bottleneck, resulting in a more lightweight model with reduced memory footprint while maintaining strong performance.
    -   **Improved SiameseTracker**: The `SiameseTracker` now fully leverages the enhanced `EfficientUNet` backbone, integrating the new attention mechanisms and improved feature fusion for more robust and accurate tracking.

-   **Aegear GUI**
    -   **Trajectory Cleanup Tools**: Introduced new features to help refine tracking data:
        -   **Highlight Outliers**: Detect and visually mark outlier points in the tracked trajectory using a configurable threshold.
        -   **Navigate Outliers**: Quickly jump between detected outlier frames for review.
        -   **Delete Outliers**: Remove individual or all detected outlier points from your tracking data.
    -   **Detailed Video Information**: A new panel in the toolbox displays comprehensive video details, including filename, FPS, resolution, total length, and frame count.
    -   **Advanced Tracking Metrics**: The application now calculates and displays both total and current travel distance of the tracked subject in centimeters, offering immediate behavioral insights.
    -   **UI Refinements**: Minor layout adjustments have been made for calibration and tracking controls for improved usability.

-   **Training Systems**
    -   **Optimized Data Loading**: Introduced `CachedDetectionDataset` and `CachedTrackingDataset` for significantly faster training data loading by pre-processing and storing image crops and metadata directly on disk.
    -   **Enhanced Negative Sampling**: Added `BackgroundWindowDataset` to generate robust negative samples (background-only image windows) from specified video segments, improving model discrimination against false positives.
    -   **Refined Heatmap Generation**: Simplified heatmap generation in datasets to directly produce Gaussian heatmaps, removing reliance on external steps.

-   **Data Caching Workflow**
    -   **New Data Preparation Notebook**: A dedicated Jupyter notebook has been introduced to automate the data caching process. This workflow handles:
        -   **Automated Data Download**: Fetches required video and annotation files from cloud storage.
        -   **Efficient Data Generation**: Creates pre-processed and augmented image crops for both detection and tracking models, including positive and negative samples.
        -   **Quality Assurance**: Provides integrated visualization tools to verify the integrity and correctness of the cached datasets before model training.

## v0.3.0

- **Tracking Improvements**
  -	Major rewrite of tracking pipeline: New adaptive frame-skipping tracking strategy,
  - Both detection (UNet) and tracking (Siamese) got significant architecture and training boost, greatly outperforming the previous system.

- **Training Systems**
  -	Unified dataset system: Introduced DetectionDataset and restructured TrackingDataset with shared design, supporting data splits, augmentation, jitter, negatives, and Gaussian heatmaps.
  - Removed contour based heatmap generation for UNet, using Gaussian instead. Relying on tracking data for training, bootstrapping previously trained tracking system for further training data mining.

- **Aegear GUI**
  - Main window layout updated: The interface is now organized with a clearer separation for the toolbox, video area, and data list. You can also resize these sections by dragging the dividers.
  - Tracking data in a table: Tracked points are now shown in a table with columns for Frame, Centroid, and Confidence, which should make them easier to look through.
  - Progress window for tracking: When you run the tracking process, a progress window will show up in the middle of the screen with status and ETA. The main window will be disabled until tracking finishes or you cancel it.


## v0.2.0

- **Tracking Improvements**
  - Enhanced trajectory drawing functionality.
  - Integrated Kalman filtering for improved tracking stability.
  - Adjusted tracking methods to reduce reliance on previous frames.

- **Model Updates**
  - Transitioned to a new Siamese tracking model for better performance.
  - Various improvements in model training have been implemented, including dataset sampling fixes and training augmentations.
  - Introduced a functional TemporalRefinementNet and Predictor Model to aid in refined output prediction.
  - Integrated trajectory prediction capabilities.

- **UI Enhancements**
  - Implemented significant UI cleanup and improvements to enhance user experience.
  - Fixed UI bindings, including adjustments for image label interactions.

### Miscellaneous Changes
- Removed dependency on moviepy to optimize performance.

## v0.1.0

Initial release of the Aegear computer vision toolkit for fish tracking and behavioral analysis in aquaculture research environments.

- aegear-gui-v0.1.0.exe – standalone Windows executable for running the GUI without needing Python.
- Built-in models for fish detection and tracking.
- Calibration tools, video loading, and tracking visualization.
