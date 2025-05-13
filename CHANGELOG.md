# Release Notes

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
