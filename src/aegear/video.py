import cv2


class VideoClip:
    """Minimalistic video clip class for reading video files."""
    def __init__(self, path):
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {path}")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.num_frames / self.fps

    def get_frame(self, t):
        """
        Return the frame at time `t` (in seconds).
        """
        frame_id = int(t * self.fps)
        return self.get_frame_by_index(frame_id)

    def get_frame_by_index(self, frame_id):
        """
        Return the frame at the given frame index.
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = self._cap.read()
        if not success:
            return None
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame
    
    def get_frame_width(self):
        """
        Return the width of the video frames.
        """
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    def get_frame_height(self):
        """
        Return the height of the video frames.
        """
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def get_frame_shape(self):
        """
        Return the shape of the video frames.
        """
        return (self.get_frame_height(), self.get_frame_width(), 3)

    def release(self):
        self._cap.release()

    def __del__(self):
        self.release()