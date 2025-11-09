#!/usr/bin/env python3
"""
Enhanced Video Trajectory Overlay Script

This script loads a video file and overlays a fading trajectory based on tracking data
from a JSON file. Features automatic ROI detection, Kalman filtering, and cropped output.
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

import cv2
import numpy as np

from aegear.utils import Kalman2D

import colorsys
from PIL import Image, ImageDraw

@dataclass
class ROI:
    """Region of Interest coordinates"""
    x: int
    y: int
    width: int
    height: int


class TrajectoryOverlay:
    def __init__(self, tracking_file: str, video_file: str, output_file: str, 
                 fade_seconds: float = 3.0, trajectory_thickness: int = 2,
                 crop_roi: bool = True, roi_margin: int = 50, process_noise: float = 1e-2, 
                 measurement_noise: float = 1e-1, skip_kalman: bool = False,
                 debug: bool = False, auto_kalman: bool = False, 
                 speed_multiplier: float = 1.0):
        """
        Initialize the trajectory overlay processor.
        
        Args:
            tracking_file: Path to JSON file containing tracking data
            video_file: Path to input video file
            output_file: Path to output video file
            fade_seconds: Number of seconds for trajectory to fade
            trajectory_thickness: Thickness of trajectory line
            crop_roi: Whether to crop video to ROI
            roi_margin: Margin around detected movement area
            process_noise: Kalman filter process noise
            measurement_noise: Kalman filter measurement noise
            skip_kalman: Skip Kalman filtering
            debug: Enable debug output
            auto_kalman: Automatically estimate Kalman parameters
            speed_multiplier: Speed up factor (2.0 = 2x speed, 0.5 = half speed)
        """
        self.tracking_file = tracking_file
        self.video_file = video_file
        self.output_file = output_file
        self.fade_seconds = fade_seconds
        self.trajectory_thickness = trajectory_thickness
        self.roi_margin = roi_margin
        self.skip_kalman = skip_kalman
        self.debug = debug
        self.speed_multiplier = speed_multiplier
        
        # Load tracking data
        self.tracking_data = self._load_tracking_data()
        self.frame_to_coords = self._build_frame_mapping()
        
        # Auto-estimate Kalman parameters if requested
        if auto_kalman and not skip_kalman:
            estimated_process_noise, estimated_measurement_noise = self._analyze_tracking_statistics()
            process_noise = estimated_process_noise
            measurement_noise = estimated_measurement_noise
        
        # Initialize Kalman filter
        if not skip_kalman:
            self.kalman_filter = Kalman2D(process_noise, measurement_noise)
        
        # Analyze ROI from tracking data
        self.roi = self._analyze_roi() if crop_roi else ROI(0, 0, 0, 0)
        
        # Apply Kalman filtering to tracking data
        if not skip_kalman:
            self._apply_kalman_filtering()
        else:
            print("Skipping Kalman filtering - using raw tracking data")
        
    def _load_tracking_data(self) -> Dict:
        """Load and parse the tracking JSON file."""
        with open(self.tracking_file, 'r') as f:
            data = json.load(f)
        return data
    
    def _build_frame_mapping(self) -> Dict[int, Tuple[int, int, float]]:
        """Build a mapping from frame_id to (x, y, confidence)."""
        frame_map = {}
        for track_point in self.tracking_data['tracking']:
            frame_id = track_point['frame_id']
            x, y = track_point['coordinates']
            confidence = track_point['confidence']
            frame_map[frame_id] = (x, y, confidence)
        return frame_map
    
    def _analyze_tracking_statistics(self) -> Tuple[float, float]:
        """
        Analyze tracking data to estimate appropriate Kalman filter parameters.
        
        Returns:
            Tuple of (process_noise, measurement_noise) estimates
        """
        print("Analyzing tracking data for automatic Kalman parameter estimation...")
        
        # Sort frames by frame_id
        sorted_frames = sorted(self.frame_to_coords.keys())
        if len(sorted_frames) < 10:
            print("Warning: Too few tracking points for reliable parameter estimation")
            return 1e-2, 1e-1
        
        # Extract positions and calculate velocities
        positions = []
        velocities = []
        accelerations = []
        measurement_errors = []
        
        for i, frame_id in enumerate(sorted_frames):
            x, y, confidence = self.frame_to_coords[frame_id]
            positions.append((x, y, confidence))
            
            # Calculate velocity (pixels per frame)
            if i > 0:
                prev_x, prev_y, _ = positions[i-1]
                frame_diff = frame_id - sorted_frames[i-1]
                if frame_diff > 0:
                    vx = (x - prev_x) / frame_diff
                    vy = (y - prev_y) / frame_diff
                    velocities.append((vx, vy))
                    
                    # Calculate acceleration (change in velocity)
                    if len(velocities) > 1:
                        prev_vx, prev_vy = velocities[-2]
                        ax = vx - prev_vx
                        ay = vy - prev_vy
                        accelerations.append((ax, ay))
            
            # Estimate measurement error based on confidence
            # Lower confidence suggests higher measurement noise
            measurement_error = (1.0 - confidence) * 10.0  # Scale factor
            measurement_errors.append(measurement_error)
        
        if not velocities or not accelerations:
            print("Warning: Insufficient data for velocity/acceleration analysis")
            return 1e-2, 1e-1
        
        # Calculate statistics
        vel_x = [vx for vx, vy in velocities]
        vel_y = [vy for vx, vy in velocities]
        acc_x = [ax for ax, ay in accelerations]
        acc_y = [ay for ax, ay in accelerations]
        
        # Velocity standard deviations (process noise related)
        vel_std_x = np.std(vel_x) if vel_x else 0
        vel_std_y = np.std(vel_y) if vel_y else 0
        vel_std = np.sqrt(vel_std_x**2 + vel_std_y**2)
        
        # Acceleration standard deviations (process noise related)
        acc_std_x = np.std(acc_x) if acc_x else 0
        acc_std_y = np.std(acc_y) if acc_y else 0
        acc_std = np.sqrt(acc_std_x**2 + acc_std_y**2)
        
        # Measurement noise estimation
        mean_measurement_error = np.mean(measurement_errors)
        
        # Window-based analysis for temporal consistency
        window_size = min(20, len(positions) // 4)  # Adaptive window size
        position_variations = []
        
        for i in range(len(positions) - window_size + 1):
            window_positions = positions[i:i + window_size]
            x_coords = [pos[0] for pos in window_positions]
            y_coords = [pos[1] for pos in window_positions]
            
            # Calculate variation within window
            x_var = np.var(x_coords) if len(x_coords) > 1 else 0
            y_var = np.var(y_coords) if len(y_coords) > 1 else 0
            position_variations.append(np.sqrt(x_var + y_var))
        
        avg_position_variation = np.mean(position_variations) if position_variations else 1.0
        
        # Estimate parameters based on analysis
        # Process noise: relates to how much we expect the motion to change
        # Higher acceleration variance suggests more dynamic motion -> higher process noise
        process_noise_base = max(acc_std * 0.1, vel_std * 0.01)
        process_noise = np.clip(process_noise_base, 1e-4, 1e-1)
        
        # Measurement noise: relates to tracking accuracy
        # Higher measurement errors and position variations suggest lower tracking quality
        measurement_noise_base = (mean_measurement_error * 0.1) + (avg_position_variation * 0.001)
        measurement_noise = np.clip(measurement_noise_base, 1e-3, 1.0)
        
        print(f"Tracking analysis results:")
        print(f"  Velocity std: {vel_std:.3f} pixels/frame")
        print(f"  Acceleration std: {acc_std:.3f} pixels/frame²")
        print(f"  Mean measurement error: {mean_measurement_error:.3f}")
        print(f"  Average position variation: {avg_position_variation:.3f}")
        print(f"  Estimated process noise: {process_noise:.6f}")
        print(f"  Estimated measurement noise: {measurement_noise:.6f}")
        
        return process_noise, measurement_noise
    
    def _analyze_roi(self) -> ROI:
        """Analyze tracking data to determine optimal ROI for cropping."""
        if not self.frame_to_coords:
            raise ValueError("No tracking data available for ROI analysis")
        
        # Extract all coordinates
        x_coords = []
        y_coords = []
        
        for x, y, confidence in self.frame_to_coords.values():
            # Weight by confidence
            if confidence > 0.5:  # Only use high-confidence detections
                x_coords.append(x)
                y_coords.append(y)
        
        if not x_coords:
            raise ValueError("No high-confidence tracking points found")
        
        # Calculate bounding box
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add margin
        roi_x = max(0, min_x - self.roi_margin)
        roi_y = max(0, min_y - self.roi_margin)
        roi_width = (max_x - min_x) + (2 * self.roi_margin)
        roi_height = (max_y - min_y) + (2 * self.roi_margin)
        
        roi = ROI(roi_x, roi_y, roi_width, roi_height)
        
        print(f"Detected ROI: x={roi.x}, y={roi.y}, width={roi.width}, height={roi.height}")
        print(f"Movement bounds: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
        
        return roi
    
    def _apply_kalman_filtering(self):
        """Apply Kalman filtering to smooth the trajectory."""
        print("Applying Kalman filtering to tracking data...")
        
        # Sort frames by frame_id
        sorted_frames = sorted(self.frame_to_coords.keys())
        
        if not sorted_frames:
            print("No frames to filter")
            return
        
        filtered_coords = {}
        
        # Debug: Print first few original coordinates
        print(f"Original coordinates (first 5):")
        for i, frame_id in enumerate(sorted_frames[:5]):
            x, y, confidence = self.frame_to_coords[frame_id]
            print(f"  Frame {frame_id}: ({x:.1f}, {y:.1f}), conf={confidence:.3f}")
        
        for i, frame_id in enumerate(sorted_frames):
            x, y, confidence = self.frame_to_coords[frame_id]
            
            # Apply Kalman filter
            filtered_x, filtered_y = self.kalman_filter.update([x, y])
            filtered_coords[frame_id] = (filtered_x, filtered_y, confidence)
            
            # Debug: Print first few filtered coordinates
            if i < 5:
                print(f"  Filtered Frame {frame_id}: ({filtered_x:.1f}, {filtered_y:.1f})")
        
        # Replace original coordinates with filtered ones
        self.frame_to_coords = filtered_coords
        print(f"Kalman filtering applied to {len(filtered_coords)} frames")
        
        # Print movement range after filtering
        if filtered_coords:
            x_coords = [x for x, _, _ in filtered_coords.values()]
            y_coords = [y for _, y, _ in filtered_coords.values()]
            print(f"Filtered movement range: x=[{min(x_coords):.1f}, {max(x_coords):.1f}], y=[{min(y_coords):.1f}, {max(y_coords):.1f}]")
    
    def _get_trajectory_points(self, current_frame: int, fps: float) -> List[Tuple[int, int, float]]:
        """
        Get trajectory points for the fade window ending at current_frame.
        Coordinates are adjusted for the ROI crop.
        
        Args:
            current_frame: Current frame number
            fps: Video frames per second
            
        Returns:
            List of (x, y, alpha) tuples where coordinates are relative to ROI
        """
        fade_frames = int(self.fade_seconds * fps)
        start_frame = max(0, current_frame - fade_frames)
        
        trajectory_points = []
        
        # Get all tracked frames in the fade window
        tracked_frames = []
        for frame_id in range(start_frame, current_frame + 1):
            if frame_id in self.frame_to_coords:
                tracked_frames.append(frame_id)
        
        if not tracked_frames:
            return trajectory_points
        
        # Calculate alpha values for fading effect
        for frame_id in tracked_frames:
            x, y, confidence = self.frame_to_coords[frame_id]
            
            # Adjust coordinates for ROI crop
            roi_x = int(x - self.roi.x)
            roi_y = int(y - self.roi.y)
            
            # Skip points outside ROI
            if self.roi.width != 0 or self.roi.height != 0:
                if roi_x < 0 or roi_y < 0 or roi_x >= self.roi.width or roi_y >= self.roi.height:
                    continue
            
            # Calculate age of this point relative to current frame
            age_frames = current_frame - frame_id
            
            # Calculate alpha based on age (1.0 = fully opaque, 0.0 = transparent)
            if fade_frames > 0:
                alpha = max(0.0, 1.0 - (age_frames / fade_frames))
            else:
                alpha = 1.0
            
            # Apply confidence weighting
            alpha *= confidence
            
            trajectory_points.append((roi_x, roi_y, alpha))
        
        return trajectory_points

    def _draw_trajectory(self, frame: np.ndarray, trajectory_points: List[Tuple[int, int, float]]) -> np.ndarray:
        
        num_points = len(trajectory_points)
        if num_points < 2:
            return frame
        
        num_segments = num_points - 1
        
        # 1. Convert OpenCV (BGR) to PIL (RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb).convert("RGBA")
        
        # 2. Create a drawing context
        trajectory_canvas = Image.new("RGBA", pil_image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(trajectory_canvas)
        
        for i in range(num_segments):
            # Normalized progress (0.0 for oldest, 1.0 for newest)
            t = (i + 1) / num_segments
            
            # Calculate color by interpolating hue
            hue = (2/3) * (1 - t)
            
            # Convert HSV (Hue, Saturation, Value) to RGB
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            
            # Scale RGB to 0-255 and set alpha
            R = int(r * 255)
            G = int(g * 255)
            B = int(b * 255)
            A = int(255 * t)
            
            # Get points
            pt1 = (int(trajectory_points[i][0]), int(trajectory_points[i][1]))
            pt2 = (int(trajectory_points[i+1][0]), int(trajectory_points[i+1][1]))
            
            # 3. Draw line with the new RGBA color
            draw.line([pt1, pt2], 
                      fill=(R, G, B, A), 
                      width=self.trajectory_thickness)
        
        # 4. Draw current position (red outline circle)
        current_x, current_y, _ = trajectory_points[-1]
        radius = 15  # New 20px radius
        bbox = [int(current_x - radius), int(current_y - radius), 
                int(current_x + radius), int(current_y + radius)]
        
        draw.ellipse(bbox, 
                     outline=(255, 0, 0, 255), 
                     width=1) 

        # 5. Composite the trajectory onto the original image
        pil_image = Image.alpha_composite(pil_image, trajectory_canvas)
        
        # 6. Convert back to OpenCV (BGR)
        frame_rgb_out = np.array(pil_image.convert("RGB"))
        output_frame = cv2.cvtColor(frame_rgb_out, cv2.COLOR_RGB2BGR)
        
        return output_frame

    def process_video(self) -> bool:
        """
        Process the video and create cropped output with trajectory overlay.
        
        Returns:
            True if successful, False otherwise
        """
        # Open input video
        cap = cv2.VideoCapture(self.video_file)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_file}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate ROI against video dimensions
        if (self.roi.x + self.roi.width > width or 
            self.roi.y + self.roi.height > height):
            print(f"Warning: ROI extends beyond video dimensions ({width}x{height})")
            # Adjust ROI to fit
            self.roi.width = min(self.roi.width, width - self.roi.x)
            self.roi.height = min(self.roi.height, height - self.roi.y)
        
        # Determine the frame range to process
        tracked_frames = sorted(self.frame_to_coords.keys())
        if not tracked_frames:
            print("Error: No tracked frames found in tracking data")
            cap.release()
            return False
        
        fade_frames = int(self.fade_seconds * fps)
        start_frame = max(0, tracked_frames[0] - fade_frames)
        end_frame = min(total_frames - 1, tracked_frames[-1])
        
        print(f"Original video: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")
        if self.roi.width > 0 and self.roi.height > 0:
            print(f"Output video: {self.roi.width}x{self.roi.height} (cropped)")
        else:
            print(f"Output video: {width}x{height} (full frame)")
        print(f"Tracked timeline: frames {tracked_frames[0]} to {tracked_frames[-1]}")
        print(f"Processing range: frames {start_frame} to {end_frame}")
        if self.speed_multiplier != 1.0:
            print(f"Speed multiplier: {self.speed_multiplier}x")
        
        # Calculate output FPS based on speed multiplier
        output_fps = fps * self.speed_multiplier
        
        # Setup video writer with cropped dimensions and adjusted FPS
        video_size = (self.roi.width, self.roi.height) if self.roi.width > 0 and self.roi.height > 0 else (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_file, fourcc, output_fps, 
                             video_size)
        
        if not out.isOpened():
            print(f"Error: Could not create output video file {self.output_file}")
            cap.release()
            return False
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames_to_process = end_frame - start_frame + 1
        frames_processed = 0
        frames_written = 0
        
        # Calculate frame skip for speed multiplier
        frame_repeat = max(1, int(1.0 / self.speed_multiplier)) if self.speed_multiplier < 1.0 else 1
        
        try:
            frame_counter = 0
            
            for current_frame in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Could not read frame {current_frame}")
                    break
                
                # Skip frames for speed multiplier > 1.0
                if self.speed_multiplier > 1.0:
                    if frame_counter % int(self.speed_multiplier) != 0:
                        frame_counter += 1
                        continue
                
                cropped_frame = frame[self.roi.y:self.roi.y + self.roi.height,
                                   self.roi.x:self.roi.x + self.roi.width] if self.roi.width > 0 and self.roi.height > 0 else frame
                
                # Get trajectory points for the ACTUAL current frame being processed
                trajectory_points = self._get_trajectory_points(current_frame, fps)

                # Draw trajectory overlay
                frame_with_trajectory = self._draw_trajectory(cropped_frame, trajectory_points)
                
                # Write frame (potentially multiple times for slow motion)
                frame_repeat = max(1, int(1.0 / self.speed_multiplier)) if self.speed_multiplier < 1.0 else 1
                for _ in range(frame_repeat):
                    out.write(frame_with_trajectory)
                    frames_written += 1
                
                frames_processed += 1
                frame_counter += 1
                
                # Progress indicator
                if frames_processed % 100 == 0 or current_frame == end_frame:
                    progress = ((current_frame - start_frame) / frames_to_process) * 100
                    print(f"\rProcessing: {progress:.1f}% (frame {current_frame}/{end_frame})", end='', flush=True)
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
            
        finally:
            # Cleanup
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        print()
        
        duration_seconds = frames_written / output_fps
        print(f"Finished processing {frames_processed} frames, wrote {frames_written} frames")
        print(f"Output duration: {duration_seconds:.1f} seconds at {output_fps:.1f} FPS")
        print(f"Cropped output saved to: {self.output_file}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Enhanced trajectory overlay with ROI analysis and Kalman filtering")
    parser.add_argument("tracking_file", help="Path to JSON tracking file")
    parser.add_argument("video_file", help="Path to input video file")
    parser.add_argument("output_file", help="Path to output video file")
    parser.add_argument("--fade-seconds", type=float, default=3.0, 
                       help="Number of seconds for trajectory to fade (default: 3.0)")
    parser.add_argument("--thickness", type=int, default=2,
                       help="Trajectory line thickness (default: 2)")
    parser.add_argument("--crop-roi", action="store_true",
                       help="Crop output video to region of interest (default: False)")
    parser.add_argument("--roi-margin", type=int, default=50,
                       help="Margin around detected movement area (default: 50)")
    parser.add_argument("--process-noise", type=float, default=1e-2,
                       help="Kalman filter process noise covariance (default: 0.01)")
    parser.add_argument("--measurement-noise", type=float, default=1e-1,
                       help="Kalman filter measurement noise covariance (default: 0.1)")
    parser.add_argument("--no-kalman", action="store_true",
                       help="Skip Kalman filtering (use raw tracking data)")
    parser.add_argument("--auto-kalman", action="store_true",
                       help="Automatically estimate Kalman parameters from tracking data")
    parser.add_argument("--speed", type=float, default=1.0,
                       help="Speed multiplier (2.0 = 2x faster, 0.5 = half speed, default: 1.0)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.tracking_file).exists():
        print(f"Error: Tracking file not found: {args.tracking_file}")
        return 1
    
    if not Path(args.video_file).exists():
        print(f"Error: Video file not found: {args.video_file}")
        return 1
    
    # Validate speed multiplier
    if args.speed <= 0:
        print("Error: Speed multiplier must be greater than 0")
        return 1
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process video
    processor = TrajectoryOverlay(
        args.tracking_file,
        args.video_file, 
        args.output_file,
        args.fade_seconds,
        args.thickness,
        args.crop_roi,
        args.roi_margin,
        args.process_noise,
        args.measurement_noise,
        args.no_kalman,
        args.debug,
        args.auto_kalman,
        args.speed
    )

    success = processor.process_video()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
