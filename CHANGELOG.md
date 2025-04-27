# Release Notes


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
