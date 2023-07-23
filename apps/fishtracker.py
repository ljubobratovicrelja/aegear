import numpy as np


class FishTracker:

    def __init__(self, clip):
        self._clip = clip
        self._track_data = {}
        self._frame_cache = {}
        pass

    def track(self, nframe):
        if nframe - 1 in self._frame_cache or nframe + 1 in self._frame_cache:
            
        pass