from typing import Tuple, Optional

import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.nn.functional import interpolate

from aegear.model import EfficientUNet, SiameseTracker


class Prediction:
    """A class to represent a prediction made by the model."""
    def __init__(self, confidence, centroid):
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
    
    def global_coordinates(self, origin):
        x, y = origin

        confidence = self.confidence
        centroid = self.centroid
        
        return Prediction(
            confidence,
            (centroid[0] + x,centroid[1] + y),
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
                 siamese_model_path,
                 tracking_threshold=0.85,
                 detection_threshold=0.95,
                 search_stride=0.5,
                 debug=False):

        self._debug = debug
        self._stride = search_stride
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = FishTracker._init_transform()
        self.heatmap_model = self._init_heatmap_model(heatmap_model_path)
        self.siamese_model = self._init_siamese_model(siamese_model_path)
        self.siamese_threshold = tracking_threshold
        self.heatmap_threshold = detection_threshold
        self.last_result = None
        self.history = []
        self.frame_size = None
    
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
    
    def track(self, frame, time, mask=None):
        if self.frame_size is None:
            self.frame_size = frame.shape[:2]

        confidence = 0.0
        self._debug_print("track")

        if self.last_result is None:
            self._debug_print("sliding")
            # Do a sliding window over the whole frame to try and find our fish.
            self.last_result = self._sliding_window_predict(frame, mask)
        else:
            self._debug_print("tracking")
            # Try getting a ROI around the last position.
            prediction, last_roi = self.last_result
            x, y = prediction.centroid

            h, w = frame.shape[:2]

            w_t = self.TRACKER_WINDOW_SIZE // 2

            # Clamp center so that full ROI fits in frame
            x = max(w_t, min(x, w - w_t))
            y = max(w_t, min(y, h - w_t))

            x1 = int(x - w_t)
            y1 = int(y - w_t)
            x2 = int(x + w_t)
            y2 = int(y + w_t)

            current_roi = frame[y1:y2, x1:x2]
            result = self._evaluate_siamese_model(last_roi, current_roi)

            if result is not None:
                self.last_result = (result.global_coordinates((x1, y1)), current_roi)
                self._debug_print(f"Found fish at ({result.centroid}) with confidence {result.confidence}")
            else:
                self.last_result = None
                self._debug_print("No fish found")

        if self.last_result is not None:
            return self.last_result[0]

        return None


    def _sliding_window_predict(self, frame, mask=None) -> Optional[Tuple[Prediction, np.array]]:
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

                self._debug_print(f"Best result at ({x}, {y}) with score {result.confidence}")

                # Map out the global coordinates of the predictions.
                global_result = result.global_coordinates((x, y))

                results.append((global_result, window))
        
        if results:
            self._debug_print(f"Got {len(results)} results")

            # Sort by score
            results.sort(key=lambda x: x[0].confidence, reverse=True)

            return results[-1]  # Return the best result

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
        output_r = interpolate(output, size=window.shape[0:2], mode='bilinear', align_corners=False)
        
        result = FishTracker._get_centroid(output_r)

        if result is None:
            self._debug_print("Heatmap: No fish detected")
            return None
        
        (confidence, centroid) = result

        
        if confidence < self.heatmap_threshold:
            self._debug_print(f"Heatmap: Confidence {confidence} is below threshold {self.heatmap_threshold}")
            return None
        
        return Prediction(confidence, centroid)
    
    def _evaluate_siamese_model(self, last_roi, current_roi) -> Prediction:

        # Prepare the input.
        template = self._transform(cv2.cvtColor(last_roi, cv2.COLOR_BGR2RGB)) \
                    .to(self._device) \
                    .unsqueeze(0)

        search = self._transform(cv2.cvtColor(current_roi, cv2.COLOR_BGR2RGB)) \
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

        if confidence < self.siamese_threshold:
            self._debug_print(f"Siamese: Confidence {confidence} is below threshold {self.siamese_threshold}")
            return None
        
        return Prediction(confidence, centroid)


    def _debug_print(self, msg):
        if self._debug:
            print(msg)