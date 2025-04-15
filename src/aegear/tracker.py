from typing import List, Tuple

import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.nn.functional import interpolate

from aegear.model import EfficientUNet, TrajectoryPredictionNet


class Prediction:
    """A class to represent a prediction made by the model."""
    def __init__(self, confidence, centroid, heatmap=None):
        """Initialize the prediction.

        Parameters
        ----------

        confidence : float
            The confidence of the prediction.
        centroid : tuple
            The centroid of the prediction.
        """

        self.centroid = centroid
        self.confidence = confidence
        self.heatmap = heatmap
    
    def global_coordinates(self, origin):
        x, y = origin

        confidence = self.confidence
        centroid = self.centroid
        
        return Prediction(
            confidence,
            (centroid[0] + x,centroid[1] + y),
            heatmap=self.heatmap
        )

class FishTracker:

    # Original window size for the training data.
    WINDOW_SIZE = 129
    # Number of max missed frames before we reset the last position.
    MAX_MISSED_FRAMES = 10
    # The size of the tracking window.
    TRACKER_WINDOW_SIZE = 129
    # The size of the trajectory prediction window.
    MAX_HISTORY_SIZE = 10

    def __init__(self,
                 heatmap_model_path,
                 trajectory_model_path,
                 tracking_threshold=0.85,
                 detection_threshold=0.95,
                 search_stride=0.5,
                 trajectory_angle_limits=(30.0, 60.0),
                 trajectory_direction_rejection_threshold=100.0,
                 debug=False):

        self._debug = debug
        self._stride = search_stride
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = FishTracker._init_transform()
        self.heatmap_model = self._init_heatmap_model(heatmap_model_path)
        self.trajectory_prediction_model = self._init_trajectory_model(trajectory_model_path)
        self.tracking_threshold = tracking_threshold
        self.detection_threshold = detection_threshold
        self.last_hit = None
        self.history = []
        self.frame_size = None
        self.trajectory_angle_limits = trajectory_angle_limits
        self.trajectory_direction_rejection_threshold = trajectory_direction_rejection_threshold
    
    def _init_transform():
        """Initialize the transform."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    def _init_heatmap_model(self, model_path):
        """Initialize the model."""
        model = EfficientUNet(weights=None)
        model.load_state_dict(torch.load(model_path, map_location=self._device))
        model.to(self._device)

        # Set the model to evaluation mode
        model.eval()
        return model
    
    def _init_trajectory_model(self, model_path):
        """Initialize the trajectory model."""
        model = TrajectoryPredictionNet()
        model.load_state_dict(torch.load(model_path, map_location=self._device))
        model.to(self._device)

        # Set the model to evaluation mode
        model.eval()
        return model
    
    def track(self, frame, time, mask=None):
        if self.frame_size is None:
            self.frame_size = frame.shape[:2]

        confidence = 0.0
        self._debug_print("track")

        if self.last_hit is None:
            self._debug_print("sliding")
            # Do a sliding window over the whole frame to try and find our fish.
            result = self._sliding_window_predict(frame, mask)
            if not result:
                return None
            
            result = result[0]  # Get the best result.
        else:
            self._debug_print("tracking")
            # Try getting a ROI around the last position.
            x, y = self.last_hit
            h, w = frame.shape[:2]

            w_t = self.TRACKER_WINDOW_SIZE // 2

            # Clamp center so that full ROI fits in frame
            x = max(w_t, min(x, w - w_t))
            y = max(w_t, min(y, h - w_t))

            x1 = int(x - w_t)
            y1 = int(y - w_t)
            x2 = int(x + w_t)
            y2 = int(y + w_t)

            roi = frame[y1:y2, x1:x2]
            result = self._evaluate_heatmap_model(roi, self.tracking_threshold)

            if not result:
                # We reset the last hit, and rerun the sliding window.
                self.last_hit = None
                return self.track(frame, mask)

            result = result.global_coordinates((x1, y1))
            
            self._debug_print(f"Found fish at ({result.centroid}) with confidence {result.confidence}")

            # Trajectory-based refinement
            refined_pos, trust_score = self._refine_with_trajectory(time, result.centroid)

            if trust_score < 0.25:
                self._debug_print("Trajectory strongly disagrees — falling back to re-detection")
                self.last_hit = None
                return self.track(frame, time, mask)
            
            self._debug_print(f"Refined position: {refined_pos} with trust score {trust_score:.2f}")

            result.centroid = refined_pos

        self.last_hit = result.centroid
        confidence = result.confidence

        # Update the history with the new prediction.
        self.history.append((time, result.centroid, result.heatmap))
        if len(self.history) > self.MAX_HISTORY_SIZE:
            self.history.pop(0)

        return (self.last_hit, confidence)


    def _sliding_window_predict(self, frame, mask=None) -> List[Prediction]:
        """
        Do a sliding window over the whole frame to try and find our fish.

        Parameters
        ----------
        frame : np.ndarray
            The frame to do the sliding window over.

        Returns
        -------

        list
            A list of predictions made by the model.
        
        """
        
        h, w = frame.shape[:2]
        results = []

        w2 = self.WINDOW_SIZE
        stride = int(self._stride * w2)

        for y in range(0, h, stride):
            for x in range(0, w, stride):

                if mask is not None:
                    mask_roi = mask[y:y+w2, x:x+w2]
                    mask_sum = mask_roi.sum()

                    # Check if the window is in the mask.
                    if mask_sum == 0:
                        continue
                
                try:
                    window = frame[y:y+w2, x:x+w2]
                except:
                    # If we go out of bounds, we skip this window.
                    continue

                if window.shape[0] != w2 or window.shape[1] != w2:
                    continue

                result = self._evaluate_heatmap_model(window, self.detection_threshold)

                if not result:
                    continue

                self._debug_print(f"Best result at ({x}, {y}) with score {result.confidence}")

                # Map out the global coordinates of the predictions.
                global_result = result.global_coordinates((x, y))

                results.append(global_result)
        
        # Sort by score
        results.sort(key=lambda x: x.confidence, reverse=True)

        return results

    def _get_centroid(heatmap):
        if heatmap.sum() < 1e-6:
            return None

        b, _, _, w = heatmap.shape
        flat_idx = torch.argmax(heatmap.view(b, -1), dim=1)
        y = flat_idx // w
        x = flat_idx % w

        # Get confidence at the centroid
        confidence = heatmap[0, 0, y, x].item()

        return confidence, (x.int().item(), y.int().item())

    def _evaluate_heatmap_model(self, window, threshold) -> Prediction:
        """Evaluate the model on a window of the image.
        Note that this returns the prediction in window local space. For global space
        adjust the centroid and box coordinates accordingly using the origin of the window.
        """

        # Prepare the input.
        input = self._transform(cv2.cvtColor(window, cv2.COLOR_BGR2RGB)) \
                    .to(self._device) \
                    .unsqueeze(0)

        try:
            output = torch.sigmoid(self.heatmap_model(input))
        except Exception as e:
            self._debug_print(f"Error in model evaluation: {e}")
            # If we get an error, we just return None.
            return None
        
        # Resize the output to the original window size.
        output_r = interpolate(output, size=(self.WINDOW_SIZE, self.WINDOW_SIZE), mode='bilinear', align_corners=False)
        
        result = FishTracker._get_centroid(output_r)

        if result is None:
            self._debug_print("No fish detected")
            if self._debug:
                h = output_r[0, 0, :, :].cpu().detach().numpy()
                cv2.circle(h, centroid, 3, (255, 0, 0), -1)
                cv2.imshow("Heatmap", h)
                cv2.imshow("Window", window)
                cv2.waitKey(1)
            return None
        
        (confidence, centroid) = result

        
        if confidence < threshold:
            self._debug_print(f"Confidence {confidence} is below threshold {threshold}")
            if self._debug:
                h = output_r[0, 0, :, :].cpu().detach().numpy()
                cv2.circle(h, centroid, 3, (255, 0, 0), -1)
                cv2.imshow("Heatmap", h)
                cv2.imshow("Window", window)
                cv2.waitKey(1)
            return None
        
        return Prediction(confidence, centroid, output[0])
    

    def _evaluate_trajectory_model(self, future_time, max_history=5):
        if len(self.history) < 2:
            return None  # Not enough history

        history = self.history[-max_history:]
        present_time, present_pos, _ = history[-1]

        print(f"Present time: {present_time}, Future time: {future_time}")

        time_horizon = future_time - present_time
        if time_horizon <= 0:
            self._debug_print(f"Ignoring GRU: non-positive horizon ({time_horizon:.3f}s)")
            return None

        rel_offsets = []
        dt_seq = []
        heatmaps = []

        for t, pos, heatmap in history:
            dt = t - present_time  # negative or zero
            dx = (pos[0] - present_pos[0]) / self.frame_size[1]
            dy = (pos[1] - present_pos[1]) / self.frame_size[0]

            dt_seq.append([dt])
            rel_offsets.append([dx, dy])
            heatmaps.append(heatmap)  # [1, H, W]

        heatmap_seq = torch.stack(heatmaps).unsqueeze(0).to(self._device)  # [1, T, 1, H, W]
        rel_offsets = torch.tensor(rel_offsets, dtype=torch.float32).unsqueeze(0).to(self._device)
        dt_seq = torch.tensor(dt_seq, dtype=torch.float32).unsqueeze(0).to(self._device)

        with torch.no_grad():
            pred = self.trajectory_prediction_model(heatmap_seq, rel_offsets, dt_seq).squeeze(0).cpu().numpy()

        # pred = [dx, dy, intensity]
        direction = pred[:2]
        intensity = pred[2]

        # Predict delta from present_pos
        predicted_delta = direction * intensity * time_horizon  # velocity * Δt

        # Scale to pixel space
        predicted_delta_px = (
            predicted_delta[0] * self.frame_size[1],
            predicted_delta[1] * self.frame_size[0]
        )

        # Add to present position (which is in pixels)
        present_px = (
            present_pos[0] * self.frame_size[1],
            present_pos[1] * self.frame_size[0]
        )

        predicted_pos_px = (
            present_px[0] + predicted_delta_px[0],
            present_px[1] + predicted_delta_px[1]
        )

        return predicted_pos_px, direction  # direction is normalized

    def _refine_with_trajectory(self, frame_time, detection_centroid):
        """
        Uses the GRU-based trajectory model to predict where the fish should be at `frame_time`,
        and compares it with the detected centroid.
        
        Returns:
            refined_centroid: adjusted or original centroid (pixels)
            trust_score: float between 0.0 and 1.0 (1.0 = full trust)
        """
        result = self._evaluate_trajectory_model(frame_time)
        if result is None:
            return detection_centroid, 1.0

        pred_pos, pred_dir = result  # direction is normalized

        # First: trust the direction at all?
        if not self._trust_trajectory_direction(pred_dir):
            self._debug_print("Trajectory predictor rejected: direction disagrees with history.")
            return detection_centroid, 0.0

        # Now compare prediction and detection
        dx = detection_centroid[0] - pred_pos[0]
        dy = detection_centroid[1] - pred_pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        trust = self._fuzzy_alignment_score(np.array([dx, dy]))

        self._debug_print(f"GRU pos: {pred_pos}, Detected: {detection_centroid}, Δ={distance:.1f}, trust={trust:.2f}")

        if trust >= 1.0:
            return detection_centroid, 1.0  # Keep heatmap

        elif trust <= 0.0:
            return None, 0.0  # Signal re-detection

        # Blend
        blended = (
            trust * detection_centroid[0] + (1 - trust) * pred_pos[0],
            trust * detection_centroid[1] + (1 - trust) * pred_pos[1]
        )
        return blended, trust


    def _trust_trajectory_prediction(self, predicted_direction: np.ndarray) -> float:
        """
        Fuzzy trust score based on angle between historical motion and GRU prediction.
        Returns trust in [0.0, 1.0].
        """

        if len(self.history) < 2:
            return 1.0

        _, pos_prev, _ = self.history[-2]
        _, pos_present, _ = self.history[-1]

        dx_hist = (pos_present[0] - pos_prev[0]) / self.frame_size[1]
        dy_hist = (pos_present[1] - pos_prev[1]) / self.frame_size[0]
        hist_vec = np.array([dx_hist, dy_hist])

        norm_hist = np.linalg.norm(hist_vec)
        norm_pred = np.linalg.norm(predicted_direction)

        if norm_hist < 1e-6 or norm_pred < 1e-6:
            return 1.0

        hist_unit = hist_vec / norm_hist
        pred_unit = predicted_direction / norm_pred

        cos_sim = np.clip(np.dot(hist_unit, pred_unit), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_sim))

        self._debug_print(f"Angle between GRU and historical motion: {angle_deg:.1f}°")

        near_limit, far_limit = self.trajectory_angle_limits

        # Fuzzy trust: full trust ≤ near threshold, linear falloff to far threshold, zero beyond
        if angle_deg <= near_limit:
            return 1.0
        elif angle_deg >= far_limit:
            return 0.0
        else:
            return 1.0 - (angle_deg - near_limit) / near_limit

    def _trust_trajectory_direction(self, predicted_direction: np.ndarray) -> bool:
        """
        Returns True if the predicted direction agrees with the recent motion history.
        """

        if len(self.history) < 2:
            return True  # No basis for rejection

        _, pos_prev, _ = self.history[-2]
        _, pos_present, _ = self.history[-1]

        dx_hist = (pos_present[0] - pos_prev[0]) / self.frame_size[1]
        dy_hist = (pos_present[1] - pos_prev[1]) / self.frame_size[0]
        hist_vec = np.array([dx_hist, dy_hist])

        norm_hist = np.linalg.norm(hist_vec)
        norm_pred = np.linalg.norm(predicted_direction)

        if norm_hist < 1e-6 or norm_pred < 1e-6:
            return True

        hist_unit = hist_vec / norm_hist
        pred_unit = np.array(predicted_direction).flatten()
        pred_unit /= norm_pred

        cos_sim = np.clip(np.dot(hist_unit, pred_unit), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_sim))

        self._debug_print(f"[Direction check] GRU vs history: {angle_deg:.1f}°")

        return angle_deg < self.trajectory_direction_rejection_threshold

    def _fuzzy_alignment_score(self, vector: np.ndarray) -> float:
        """
        Fuzzy trust score based on alignment between GRU-predicted and heatmap-based positions.
        Assumes 'vector' is (heatmap_pos - pred_pos).
        """

        if len(vector) != 2:
            return 0.0

        norm = np.linalg.norm(vector)
        if norm < 1e-6:
            return 1.0

        direction = vector / norm
        return self._trust_trajectory_prediction(direction)  # same fuzzy falloff logic

    def _debug_print(self, msg):
        if self._debug:
            print(msg)