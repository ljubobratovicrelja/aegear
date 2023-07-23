import cv2
import numpy as np


class MotionDetector:

    def __init__(self, threshold=0.3, block_size=32):
        self.threshold = threshold
        self.block_size = block_size
    
    def _mad(self, block1, block2):
        return np.mean(np.abs(block1 - block2))

    def detect(self, previous, current):
        gframe = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        gprevFrame = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)

        if gframe.dtype == np.uint8:
            gframe = gframe.astype(np.float32) / 255.0
            gprevFrame = gprevFrame.astype(np.float32) / 255.0
        elif gframe.dtype == np.uint16:
            gframe = gframe.astype(np.float32) / 65535.0
            gprevFrame = gprevFrame.astype(np.float32) / 65535.0

        num_rows_blocks = int(np.floor(gframe.shape[0] / self.block_size))
        num_cols_blocks = int(np.floor(gframe.shape[1] / self.block_size))

        result = np.zeros((num_rows_blocks, num_cols_blocks), dtype=np.uint8)

        for r in range(0, num_rows_blocks):
            for c in range(0, num_cols_blocks):
                row = r * self.block_size
                col = c * self.block_size

                if row + self.block_size > gframe.shape[0] or col + self.block_size > gframe.shape[1]:
                    continue

                block = gframe[row:row + self.block_size, col:col + self.block_size]
                prevBlock = gprevFrame[row:row + self.block_size, col:col + self.block_size]

                mad = self._mad(block, prevBlock)

                if mad > self.threshold:
                    result[r, c] = 255
        
        return result