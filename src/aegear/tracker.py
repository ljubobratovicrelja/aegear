from typing import Tuple, Optional

import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.nn.functional import interpolate

from aegear.model import EfficientUNet, SiameseTracker
from aegear.video import VideoClip
from aegear.gui.progress_reporter import ProgressReporter


class Prediction:
    """A class to represent a prediction made by the model."""
    def __init__(self, confidence, centroid, roi=None):
        """Initialize the prediction.

        Parameters
        ----------

        confidence : float
            The confidence of the prediction.
        centroid : tuple
            The centroid of the prediction.
        roi : np.ndarray
            The region of interest of the prediction.
        """

        self.centroid = centroid
        self.confidence = confidence
        self.roi = roi
    
    def global_coordinates(self, origin):
        x, y = origin

        confidence = self.confidence
        centroid = self.centroid
        
        return Prediction(
            confidence,
            (centroid[0] + x,centroid[1] + y),
            self.roi,
        )

class Kalman2D:
    def __init__(self, r=1.0, q=0.1):
        """Initialize the Kalman filter.
        
        Parameters
        ----------
        r : float
            The measurement noise.
        q : float
            The process noise.
        """
        self.x = np.zeros((4, 1))  # state
        self.P = np.eye(4) * 1000  # uncertainty

        self.A = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        self.R = np.eye(2) * r # measurement noise
        self.Q = np.eye(4) * q # process noise

    def reset(self, x, y):
        self.x = np.array([[x], [y], [0], [0]])
        self.P = np.eye(4)

    def update(self, z):
        # Predict
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

        # Update
        z = np.array(z).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.x[0, 0], self.x[1, 0]

class FishTracker:

    # Original window size for the training data.
    WINDOW_SIZE = 129
    # The size of the tracking window.
    TRACKER_WINDOW_SIZE = 129

    def __init__(self,
                 heatmap_model_path,
                 siamese_model_path,
                 tracking_threshold=0.9,
                 detection_threshold=0.85,
                 search_stride=0.5,
                 tracking_max_skip=10,
                 debug=False):

        self._debug = debug
        self._stride = search_stride
        self._device = FishTracker._select_device()
        self._transform = FishTracker._init_transform()
        self.heatmap_model = self._init_heatmap_model(heatmap_model_path)
        self.siamese_model = self._init_siamese_model(siamese_model_path)
        self.tracking_threshold = tracking_threshold
        self.detection_threshold = detection_threshold
        self.tracking_max_skip = tracking_max_skip

        self.last_result = None
        self.history = []
        self.frame_size = None
        self.kalman = Kalman2D()
    
    def run_tracking(self, video: VideoClip, start_frame: int, end_frame: int, progress_reporter: ProgressReporter, model_track_register, ui_update):
        """Run the tracking on a video."""
      
        bgs = self._init_background_subtractor(video, start_frame)
        current_skip = self.tracking_max_skip
        anchor_frame = start_frame

        self.last_result = None

        while anchor_frame < end_frame and progress_reporter.still_running():
            candidate = anchor_frame + current_skip
            if candidate >= end_frame:
                break

            # Read and pre‑process the candidate.
            frame = video.get_frame(float(candidate) / video.fps)
            if frame is None:
                break

            result = self._track_frame(frame, mask=self._motion_detection(bgs, frame))

            if result is not None:
                # Store this result for further tracking.
                self.last_result = result
                model_track_register(candidate, result.centroid, result.confidence)

                anchor_frame = candidate
                progress_reporter.update(anchor_frame)

                if current_skip < self.tracking_max_skip:
                    current_skip = min(current_skip * 2, self.tracking_max_skip)
            else:
                if self.last_result is not None and current_skip > 1:
                    current_skip = max(current_skip // 2, 1)
                    continue

                anchor_frame = candidate
                self.last_result = None

            ui_update(anchor_frame)

    def _select_device():
        """Select the device - try CUDA, if fails, try mps for Apple Silicon, else CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
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
    
    def _init_siamese_model(self, model_path):
        """Initialize the siamese tracking model."""
        model = SiameseTracker()
        model.load_state_dict(torch.load(model_path, map_location=self._device))
        model.to(self._device)

        # Set the model to evaluation mode
        model.eval()
        return model
    
    def _track_frame(self, frame, mask=None):
        """Track the fish in the given frame.
        
        Parameters
        ----------
        
        frame : np.ndarray
            The frame to track the fish in.
        mask : np.ndarray, optional
            The mask to use for tracking. If None, the whole frame is used.

        Returns
        -------

        Prediction or None
            The prediction made by the model, or None if no fish is detected.
        """
        if self.frame_size is None:
            self.frame_size = frame.shape[:2]

        self._debug_print("track")

        if self.last_result is None:
            self._debug_print("sliding")
            # Do a sliding window over the whole frame to try and find our fish.
            result = self._sliding_window_predict(frame, mask)

            if result is not None:
                prediction = result

                prediction.roi = self._tracking_roi(frame, prediction.centroid)[1]

                self.kalman.reset(prediction.centroid[0], prediction.centroid[1])

                return prediction
        else:
            self._debug_print("tracking")
            # Try getting a ROI around the last position.
            (x1, y1), current_roi = self._tracking_roi(frame, self.last_result.centroid)
            result = self._evaluate_siamese_model(self.last_result.roi, current_roi)

            if result is not None:
                prediction = result.global_coordinates((x1, y1))

                x, y = self.kalman.update(prediction.centroid)
                prediction.centroid = (int(x), int(y))

                prediction.roi = self._tracking_roi(frame, prediction.centroid)[1]

                self._debug_print(f"Found fish at ({result.centroid}) with confidence {result.confidence}")

                return prediction

        return None
    
    def _tracking_roi(self, frame, centroid):
        """Get the tracking ROI around the centroid."""
        x, y = centroid
        h, w = frame.shape[:2]
        w_t = self.TRACKER_WINDOW_SIZE // 2

        # Clamp center so that full ROI fits in frame
        x = max(w_t, min(x, w - w_t))
        y = max(w_t, min(y, h - w_t))

        x1 = int(x - w_t)
        y1 = int(y - w_t)
        x2 = int(x + w_t)
        y2 = int(y + w_t)

        return (x1, y1), frame[y1:y2, x1:x2]
    
    def _init_background_subtractor(self, video: VideoClip, start_frame: int, history=50, dist2threshold=500, warmup=20):
        """Initialize the background subtractor."""
        background_subtractor = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=dist2threshold, detectShadows=False)

        # Warm up the background subtractor with a few frames.
        for fid in range(max(start_frame - warmup, 0), start_frame):
            t = float(fid) / video.fps
            f = video.get_frame(t)
            if f is None:
                continue

            gframe = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            gframe = cv2.GaussianBlur(gframe, (5, 5), 1.0)

            background_subtractor.apply(gframe, learningRate=0.25)
        
        return background_subtractor
    
    def _motion_detection(self, bgs, frame):
        """Detect motion in the frame using the background subtractor."""

        gframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gframe = cv2.GaussianBlur(gframe, (5, 5), 1.0)

        mask = bgs.apply(gframe, learningRate=0.125)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        return mask

    def _sliding_window_predict(self, frame, mask=None) -> Optional[Prediction]:
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

        win_size = self.WINDOW_SIZE
        stride = int(self._stride * win_size)

        for y in range(0, h, stride):
            for x in range(0, w, stride):

                if mask is not None:
                    mask_roi = mask[y:y+win_size, x:x+win_size]
                    mask_sum = mask_roi.sum()

                    # Check if the window is in the mask.
                    if mask_sum == 0:
                        continue
                
                try:
                    window = frame[y:y+win_size, x:x+win_size]
                except:
                    # If we go out of bounds, we skip this window.
                    continue

                if window.shape[0] != win_size or window.shape[1] != win_size:
                    continue

                result = self._evaluate_heatmap_model(window)

                if not result:
                    continue

                # Map out the global coordinates of the predictions.
                results.append(result.global_coordinates((x, y)))
        
        if results:
            self._debug_print(f"Got {len(results)} results")

            # Sort by score
            results.sort(key=lambda x: x.confidence, reverse=True)

            # Get the best result
            result = results[0]
        
            if result.confidence < self.detection_threshold:

                self._debug_print(f"Best candidate confidence {result.confidence} is below threshold {self.detection_threshold}")
                return None

            return result  # Return the best result

        self._debug_print(f"Not a single sliding window found a fish")

        return None

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
    
    def _evaluate_heatmap_model(self, window) -> Prediction:
        """Evaluate the model on a window of the image.
        Note that this returns the prediction in window local space. For global space
        adjust the centroid and box coordinates accordingly using the origin of the window.
        """

        # Prepare the input.
        input = self._transform(window) \
                    .to(self._device) \
                    .unsqueeze(0)

        try:
            output = torch.sigmoid(self.heatmap_model(input))
        except Exception as e:
            self._debug_print(f"Error in model evaluation: {e}")
            # If we get an error, we just return None.
            return None
        
        # Resize the output to the original window size.
        output_r = interpolate(output, size=window.shape[0:2], mode='bilinear', align_corners=False)
        
        result = FishTracker._get_centroid(output_r)

        if result is None:
            self._debug_print("Heatmap: No fish detected")
            return None
        
        (confidence, centroid) = result
        
        return Prediction(confidence, centroid)
    
    def _evaluate_siamese_model(self, last_roi, current_roi) -> Prediction:

        # Prepare the input.
        template = self._transform(last_roi) \
                    .to(self._device) \
                    .unsqueeze(0)

        search = self._transform(current_roi) \
                    .to(self._device) \
                    .unsqueeze(0)

        try:
            output = torch.sigmoid(self.siamese_model(template, search))
        except Exception as e:
            self._debug_print(f"Siamese: Error in model evaluation: {e}")
            # If we get an error, we just return None.
            return None

        # Resize the output to the original window size.
        output_r = interpolate(output, size=(self.WINDOW_SIZE, self.WINDOW_SIZE), mode='bilinear', align_corners=False)

        result = FishTracker._get_centroid(output_r)

        if result is None:
            self._debug_print("Siamese: No fish detected")
            return None
        
        (confidence, centroid) = result

        if  confidence < self.tracking_threshold:
            self._debug_print(f"Siamese: Confidence {confidence} is below threshold {self.tracking_threshold}")
            return None
        
        # Note that for siamese we don't store the roi, because we afterwards do kalman filtering.
        return Prediction(confidence, centroid, roi=None)


    def _debug_print(self, msg):
        if self._debug:
            print(msg)