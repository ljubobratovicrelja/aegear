# Usage

This guide explains how to use the Aegear GUI to perform fish tracking and trajectory analysis in controlled aquatic environments. The application is designed for offline analysis and provides a user-friendly interface for video loading, calibration, tracking, and data review.

---

## 🏁 Quick Start Workflow

1. **Launch Aegear GUI**
    - Run the prebuilt binary (Windows/macOS) or use the command (within the Python virtual environment and requirements installed): `aegear-gui`

2. **Load Video**
    - File → *Load Video…* (or simply left click on the central panel with Aegear logo)
    - Select a video file (AVI, MP4, MKV, MOV, WMV, FLV, MPEG).
    - The video will appear in the main display area with playback controls enabled.

3. **(Optional) Load Calibration**
     - File → *Load Scene Reference…* (for metric scaling)
     - Calibrate the scene for the loaded video by starting the calibration procedure by clicking the *Calibrate* button.
     - Select 4 corners within the video that represent the scene calibration (reference) points.

4. **Run Tracking**
     - Navigate to the point in the video where tracking should start and click *Set Track Start*.
     - Similarly, go to a point in the video where tracking should end and click *Set Track End*.
     - Click *Run Tracking* to start detection and tracking.
     - A progress window pops up and displays the estimated time for completion.

5. **Review & Edit Trajectories**
     - Use the **treeview** on the right to navigate tracked frames.
     - Navigate between tracked frames using arrow keys.
     - Left click on the video panel to create or adjust a tracked point.
     - Right click to remove the tracked point at the current frame.
     - Highlight and clean outliers using the outlier tools.

6. **Save Results**
     - File → *Save Tracking…* to export results in JSON format.

---

## 🎬 Video Playback Controls

| Control               | Description                           |
|-----------------------|---------------------------------------|
| ▶️ Play               | Play video playback                  |
| ⏸️ Pause              | Pause playback                       |
| ⏮ Previous Frame      | Step backward one frame              |
| ⏭ Next Frame          | Step forward one frame               |
| 🎚 Slider             | Scrub through video frames           |
| 📃 Timeline Label     | Displays current frame and timestamp |

---

## 🔬 Calibration

Calibration is optional but recommended for accurate metric measurements. For detailed calibration instructions, please
see the [calibration](calibration.md) page.

## 🧠 Tracking and Settings Explained

Aegear’s tracking relies on deep learning models from the backend:

### Detection
- **Backend**: U-Net variant (EfficientUNet) with EfficientNet-B0 encoder.
- **UI Setting: Detection Threshold**
    - Adjusts minimum confidence for fish detection heatmaps.
    - Higher values reduce false positives but may miss subtle detections.

### Tracking
- **Backend**: Siamese network for frame-to-frame association.
- **UI Setting: Tracking Threshold**
    - Sets confidence threshold for the response map.
    - If confidence drops below this, the system falls back to sliding-window detection.

### Trajectory and Smart Frame Skip
- **UI Setting: Trajectory Frame Skip**
    - Controls how often trajectory points are plotted in the overlay for visualization.
    - Higher values thin out visual clutter for long videos.

- **Adaptive Frame Skipping During Tracking**
    - Aegear also uses a smart frame skipping mechanism for performance during tracking:
      1. Begins with the user-defined frame skip (N), processing only every Nth frame.
      2. If tracking fails in a skipped frame, automatically retries with half the skip (N/2), continuing recursively until N=1 (no skipping).
      3. Once tracking confidence recovers, reverts to the original N.
    - This system optimizes speed on easy sequences while preserving accuracy in complex sections.
    - Fully transparent to the user, this logic works behind the scenes using the same "Frame Skip" parameter.

---

## ⚠️ Outlier Detection & Cleanup

| Tool                         | Description                                    |
|------------------------------|------------------------------------------------|
| Highlight Outliers           | Marks suspect trajectory points (velocity jumps) |
| Next/Previous Outlier        | Navigate between flagged outliers              |
| Delete Current Outlier       | Removes selected outlier from tracking data    |
| Delete All Outliers          | Clears all flagged outliers                    |

Backend: Uses `detect_trajectory_outliers` for identifying anomalies.

---

## 💾 Saving and Loading

| Action                  | Menu Path                 |
|-------------------------|---------------------------|
| Save Tracking Data      | File → *Save Tracking…*   |
| Load Tracking Data      | File → *Load Tracking…*   |
| Exit Application        | File → *Exit*             |

---


## 📝 Notes

- Prebuilt binaries include CPU-only PyTorch.
- For GPU acceleration, install Aegear from source and use a CUDA-enabled PyTorch build.