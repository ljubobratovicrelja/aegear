from typing import List

import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo


class Prediction:
    """A class to represent a prediction made by the model."""
    def __init__(self, box, score, centroid):
        """
        Parameters
        ----------
        box : list
            A list of 4 integers representing the bounding box of the fish.
        score : float
            A float representing the score of the prediction.
        centroid : tuple
            A tuple of 2 floats representing the centroid of the fish in global frame coordinates.
        """

        self.box = box
        self.score = score
        self.centroid = centroid
    
    def global_coordinates(self, origin):
        x, y = origin

        box = self.box
        centroid = self.centroid
        score = self.score
        
        return Prediction(
            box=[box[0] + x, box[1] + y, box[2] + x, box[3] + y],
            score=score,
            centroid=(centroid[0] + x,centroid[1] + y)
        )

class FishTracker:

    # Original window size for the training data.
    TRAINING_WINDOW_SIZE = 129
    # Number of max missed frames before we reset the last position.
    MAX_MISSED_FRAMES = 10
    # Tracker window size.
    TRACKER_WINDOW_SIZE = 64

    """A class to track fish in a video stream."""

    def __init__(self, model_path, score_threshold=0.5, tracking_threshold=0.75, detection_threshold=0.95, tracking_window=256, window_size=(256, 512), window_stride=256, debug=False):
        """
        Parameters
        ----------

        model_path : str
                The path to the model weights.
        score_threshold : float, optional
                The score threshold for the model. The default is 0.75.
        detection_threshold : float, optional
                    The detection threshold for the model. The default is 0.9.
        window_size : tuple, optional
                The size of the sliding window. The default is (256, 512).
        window_stride : int, optional
                    The stride of the sliding window. The default is 256.   
        """

        self.window_size = window_size
        self.tracking_threshold = tracking_threshold
        self.tracking_window = tracking_window
        self.window_stride = window_stride
        self._model = FishTracker._init_model(model_path, score_threshold, window_size)
        self.detection_threshold = detection_threshold
        self.last_hit = None
        self.missed_frames = 0
        self.tracker = None
        self._debug = debug
    
    def _init_model(model_path,score_threshold, window_size):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
            #"COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
        ))

        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.INPUT.MIN_SIZE_TEST = window_size[0]
        cfg.INPUT.MAX_SIZE_TEST = window_size[1]

        return DefaultPredictor(cfg)
    
    def track(self, frame, mask=None):
        confidence = 0.0

        self._debug_print("track")

        if self.last_hit is None or self.missed_frames > FishTracker.MAX_MISSED_FRAMES:
            self._debug_print("sliding")
            # Do a sliding window over the whole frame to try and find our fish.
            results = self._sliding_window_predict(frame, mask)
            if not results:
                return None
            
            self.last_hit = results[0].centroid
            confidence = results[0].score
            self.missed_frames = 0

            # Initialize the tracker with the first hit.
            self._reinit_tracker(frame, self.last_hit)
        else:
            self._debug_print("tracking")
            # Try getting a ROI around the last position.
            x, y = self.last_hit
            h, w = frame.shape[:2]

            w_t = self.tracking_window // 2

            # Clamp center so that full ROI fits in frame
            x = max(w_t, min(x, w - w_t))
            y = max(w_t, min(y, h - w_t))

            x1 = int(x - w_t)
            y1 = int(y - w_t)
            x2 = int(x + w_t)
            y2 = int(y + w_t)

            roi = frame[y1:y2, x1:x2]
            results = self._evaluate_model(roi, self.tracking_threshold)

            if not results:
                # If we don't find anything, we reset the last position and try again.
                self.missed_frames += 1
                self._debug_print("Missed tracking frame")

                success, box = self.tracker.update(frame)
                if success:
                    # Update the last hit with the new position.
                    x1, y1, w, h = [int(v) for v in box]
                    x = int(x1 + w / 2)
                    y = int(y1 + h / 2)

                    self.last_hit = (x, y)
                    confidence = 0.0

                    if self._debug:
                        draw_frame = frame.copy()
                        cv2.rectangle(draw_frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                        cv2.imshow("Interpolation", draw_frame)
                        cv2.waitKey(1)

                    return (self.last_hit, confidence)
                else:
                    return None
            
            best_result = results[0].global_coordinates((x1, y1))
            self._debug_print(f"Found fish at ({best_result.centroid}) with score {best_result.score}")

            self.missed_frames = 0

            self.last_hit = best_result.centroid
            confidence = best_result.score

            self._reinit_tracker(frame, self.last_hit)

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

        for y in range(0, h, self.window_stride):
            for x in range(0, w, self.window_stride):

                if mask is not None:
                    mask_roi = mask[y:y+self.window_size[1], x:x+self.window_size[0]]
                    mask_sum = mask_roi.sum()

                    # Check if the window is in the mask.
                    if mask_sum == 0:
                        continue
                
                try:
                    window = frame[y:y+self.window_size[1], x:x+self.window_size[0]]
                except:
                    # If we go out of bounds, we skip this window.
                    continue

                if window.shape[0] != self.window_size[1] or window.shape[1] != self.window_size[0]:
                    continue

                window_results = self._evaluate_model(window, self.detection_threshold)

                if not window_results:
                    continue

                # Do thorough check on the results to confirm our result.
                #window_results = self._thorough_check(window_results, window)

                #if not window_results:
                    #continue

                # Data is already sorted by score so just take the first one, as we're only
                # interested in the best score per sliding window.
                best_result = window_results[0]

                self._debug_print(f"Best result at ({x}, {y}) with score {best_result.score}")

                # Map out the global coordinates of the predictions.
                global_result = best_result.global_coordinates((x, y))

                results.append(global_result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)

        return results
    
    def _evaluate_model(self, window, threshold) -> List[Prediction]:
        """Evaluate the model on a window of the image.
        Note that this returns the prediction in window local space. For global space
        adjust the centroid and box coordinates accordingly using the origin of the window.
        """

        try:
            outputs = self._model(window)
            instances = outputs["instances"].to("cpu")
        except Exception as e:
            self._debug_print(f"Error in model evaluation: {e}")
            # If we get an error, we just return None.
            return None

        results = []

        for i in range(len(instances)):
            score = instances.scores[i].item()
            if score > threshold:

                box = instances.pred_boxes[i].tensor.numpy()[0]
                mask = instances.pred_masks[i].numpy().astype(np.uint8)

                x1, y1, x2, y2 = box

                # Get centroid of the mask
                centroid = None
                moments = cv2.moments(mask)
                if moments["m00"] != 0:
                    centroid = (
                        int(moments["m10"] / moments["m00"]),
                        int(moments["m01"] / moments["m00"])
                    )
                else:
                    # Fallback to the center of the bbox
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                #centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                results.append(Prediction(box, score, centroid))
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _thorough_check(self, cross_check_results, frame, iterations=1):
        """
        For each of the results, we do a check where we take the window around the
        hit, and rotate it a bit to confirm our result.

        Parameters
        ----------
        results : list
            A list of predictions made by the model.
        frame : np.ndarray
            The frame to do the sliding window over.
        iterations : int, optional
            The number of iterations to do the thorough check. The default is 3.

        Returns
        -------

        list
            A list of predictions made by the model.
        
        """
        
        good_results = []

        for result in cross_check_results:
            
            # Sample the roi using the center and training window size.
            x, y = result.centroid
            w2 = FishTracker.TRAINING_WINDOW_SIZE // 2 + 5 # Add a bit of padding

            x1 = int(x - w2)
            y1 = int(y - w2)

            x2 = int(x + w2)
            y2 = int(y + w2)

            try:
                roi = frame[y1:y2, x1:x2]
            except:
                # If we go out of bounds, we skip this result.
                continue

            # Check if the roi is valid.
            if roi.shape[0] * roi.shape[1] == 0:
                continue

            passed = True
            for _ in range(iterations):
                # Construct a random rotation matrix
                angle = np.random.uniform(-15, 15)
                center = (roi.shape[1] // 2, roi.shape[0] // 2)

                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_roi = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]))

                cross_check_results = self._evaluate_model(rotated_roi)

                if not cross_check_results:
                    continue

                best_result = cross_check_results[0]

                # We check out the score, and if its above the detection threshold, we add it to the results.
                if best_result.score < self.detection_threshold:
                    passed = False
                    break

            if passed:
                # It has passed the test, just add the result to the good results.
                good_results.append(result)

        return good_results
    
    def _reinit_tracker(self, frame, hit): 
        x, y = hit
        w = FishTracker.TRACKER_WINDOW_SIZE // 2
        bbox = [int(x-w), int(y-w), int(w*2), int(w*2)]

        self.tracker = cv2.legacy.TrackerMOSSE_create()
        self.tracker.init(frame, bbox)

        if self._debug:
            draw_frame = frame.copy()
            cv2.rectangle(draw_frame, (int(x-w), int(y-w)), (int(x+w), int(y+w)), (0, 255, 0), 2)

            cv2.imshow("Init", draw_frame)
            cv2.waitKey(1)
    
    def _debug_print(self, msg):
        if self._debug:
            print(msg)