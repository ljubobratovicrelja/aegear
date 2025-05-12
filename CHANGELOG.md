# Release Notes

## v0.3.0

-	Major rewrite of tracking pipeline: New adaptive frame-skipping tracker using heatmap + Siamese fusion, with robust background subtraction and confidence-based control. Tracking loop now fully decoupled and externally controlled via run_tracking().
-	GUI tracking overhaul: Replaced static loop with responsive UI integration and live progress feedback (ProgressReporter), exposing threshold controls and skip stride interactively.
-	Unified dataset system: Introduced DetectionDataset and restructured TrackingDataset with shared design, supporting data splits, augmentation, jitter, negatives, and Gaussian heatmaps.
-	Model architecture refinement: Removed unused decoder depth; simplified upsampling blocks using bilinear interpolation and batch norm; Siamese tracker now reuses encoder blocks and merges features more effectively.
-	Calibration update: Added precise point rectification method to support accurate coordinate mapping.

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
