"""
Block Matching Algorithm

Module to detect motion between two frames using the block matching algorithm.
"""

import numpy as np


class BlockMatching:

    def __init__(self, block_size, threshold):
        self.block_size = block_size
        self.threshold = threshold

    def compute_sad(self, block1, block2):
        return np.sum(np.abs(block1-block2))

    def normalize_img(self, img):
        return (img - np.min(img)) / (np.max(img) - np.min(img)).astype(np.float32)

    def detect_changes(self, img1, img2):
        img1 = self.normalize_img(img1)
        img2 = self.normalize_img(img2)
        
        h, w = img1.shape
        change_map = np.zeros((h//self.block_size, w//self.block_size), dtype=np.uint8)

        for i in range(0, h-self.block_size, self.block_size):
            for j in range(0, w-self.block_size, self.block_size):
                sad = self.compute_sad(img1[i:i+self.block_size, j:j+self.block_size], img2[i:i+self.block_size, j:j+self.block_size])
                if sad > self.threshold:
                    change_map[i//self.block_size, j//self.block_size] = 1

        return change_map
