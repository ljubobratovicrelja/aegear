from typing import List

import cv2

import torch
import torchvision.transforms as transforms
from torch.nn.functional import interpolate

from aegear.model import EfficientUNet


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
            (centroid[0] + x,centroid[1] + y)
        )

class FishTracker:

    # Original window size for the training data.
    WINDOW_SIZE = 129
    # Number of max missed frames before we reset the last position.
    MAX_MISSED_FRAMES = 10
    # The size of the tracking window.
    TRACKER_WINDOW_SIZE = 129

    def __init__(self, model_path, tracking_threshold=0.85, detection_threshold=0.95, search_stride=0.5, debug=False):
        self._debug = debug
        self._stride = search_stride
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = FishTracker._init_transform()
        self.model = self._init_model(model_path)
        self.tracking_threshold = tracking_threshold
        self.detection_threshold = detection_threshold
        self.last_hit = None
    
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

    def _init_model(self, model_path):
        """Initialize the model."""
        model = EfficientUNet(weights=None)
        model.load_state_dict(torch.load(model_path, map_location=self._device))
        model.to(self._device)

        # Set the model to evaluation mode
        model.eval()
        return model
    
    def track(self, frame, mask=None):
        confidence = 0.0
        self._debug_print("track")

        if self.last_hit is None:
            self._debug_print("sliding")
            # Do a sliding window over the whole frame to try and find our fish.
            result = self._sliding_window_predict(frame, mask)
            if not result:
                return None
            
            self.last_hit = result[0].centroid
            confidence = result[0].confidence
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
            result = self._evaluate_model(roi, self.tracking_threshold)

            if not result:
                # We reset the last hit, and rerun the sliding window.
                self.last_hit = None
                return self.track(frame, mask)

            result = result.global_coordinates((x1, y1))
            
            self._debug_print(f"Found fish at ({result.centroid}) with confidence {result.confidence}")

            self.last_hit = result.centroid
            confidence = result.confidence

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

                result = self._evaluate_model(window, self.detection_threshold)

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

    def _evaluate_model(self, window, threshold) -> Prediction:
        """Evaluate the model on a window of the image.
        Note that this returns the prediction in window local space. For global space
        adjust the centroid and box coordinates accordingly using the origin of the window.
        """

        # Prepare the input.
        input = self._transform(cv2.cvtColor(window, cv2.COLOR_BGR2RGB)) \
                    .to(self._device) \
                    .unsqueeze(0)

        try:
            output = torch.sigmoid(self.model(input))
        except Exception as e:
            self._debug_print(f"Error in model evaluation: {e}")
            # If we get an error, we just return None.
            return None
        
        # Resize the output to the original window size.
        output = interpolate(output, size=(self.WINDOW_SIZE, self.WINDOW_SIZE), mode='bilinear', align_corners=False)
        
        result = FishTracker._get_centroid(output)

        if result is None:
            self._debug_print("No fish detected")
            if self._debug:
                h = output[0, 0, :, :].cpu().detach().numpy()
                cv2.circle(h, centroid, 3, (255, 0, 0), -1)
                cv2.imshow("Heatmap", h)
                cv2.imshow("Window", window)
                cv2.waitKey(1)
            return None
        
        (confidence, centroid) = result

        
        if confidence < threshold:
            self._debug_print(f"Confidence {confidence} is below threshold {threshold}")
            if self._debug:
                h = output[0, 0, :, :].cpu().detach().numpy()
                cv2.circle(h, centroid, 3, (255, 0, 0), -1)
                cv2.imshow("Heatmap", h)
                cv2.imshow("Window", window)
                cv2.waitKey(1)
            return None
        
        return Prediction(confidence, centroid)
    
    def _debug_print(self, msg):
        if self._debug:
            print(msg)