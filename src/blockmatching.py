"""
Block Matching Algorithm

Module to detect motion between two frames using the block matching algorithm.
"""


import cv2

import numpy as np


class BlockMatching:
    """
    Class to detect motion between two frames using the block matching algorithm.
    """

    def __init__(self, threshold=0.5, blockSize=16, searchRange=16):
        """
        Constructor.

        :param threshold: Threshold to determine if there is motion in a block.
        :param blockSize: Size of the blocks to be compared.
        :param searchRange: Search range for the block matching algorithm.
        """

        self.threshold = threshold
        self.blockSize = blockSize
        self.searchRange = searchRange

    def _hasAnyMotion(self, frame1, frame2, threshold=0.2):
        frame1 = cv2.resize(frame1, (0, 0), fx=0.125, fy=0.125)
        frame2 = cv2.resize(frame2, (0, 0), fx=0.125, fy=0.125)

        return np.mean(frame1 - frame2) > threshold

    def detect(self, frame1, frame2):
        """
        Detect motion between two frames.

        :param frame1: First frame.
        :param frame2: Second frame.
        :return: Motion mask.
        """
        
        # get frame dimensions
        height, width = frame1.shape

        # get number of blocks
        numBlocksY = int(height / self.blockSize)
        numBlocksX = int(width / self.blockSize)

        # initialize motion mask
        motionMask = np.zeros((height, width), dtype=np.uint8)

        if not self._hasAnyMotion(frame1, frame2):
            return motionMask

        # iterate through blocks
        for by in range(numBlocksY):
            for bx in range(numBlocksX):
                # get block
                block1 = frame1[by * self.blockSize:(by + 1) * self.blockSize,
                         bx * self.blockSize:(bx + 1) * self.blockSize]
                block2 = frame2[by * self.blockSize:(by + 1) * self.blockSize,
                         bx * self.blockSize:(bx + 1) * self.blockSize]

                # get motion vector
                motionVector = self._getMotionVector(block1, block2)

                # calculate motion magnitude
                magnitude = np.linalg.norm(motionVector)

                # check if magnitude is greater than threshold
                if magnitude > self.threshold:
                    # mark block as motion
                    motionMask[by * self.blockSize:(by + 1) * self.blockSize,
                    bx * self.blockSize:(bx + 1) * self.blockSize] = 255

        return motionMask
    
    def _getMotionVector(self, block1, block2):
        """
        Get motion vector between two blocks.

        :param block1: First block.
        :param block2: Second block.
        :return: Motion vector.
        """

        # get block dimensions
        height, width = block1.shape

        # initialize best motion vector
        bestMotionVector = np.array([0, 0])

        # initialize best error
        bestError = np.inf

        # iterate through search range
        for dy in range(-self.searchRange, self.searchRange):
            for dx in range(-self.searchRange, self.searchRange):
                # get motion vector
                motionVector = np.array([dx, dy])

                # get block
                block = self._getBlock(block2, motionVector, height, width)

                # calculate error
                error = np.sum(np.abs(block1 - block))

                # check if error is less than best error
                if error < bestError:
                    # update best error
                    bestError = error

                    # update best motion vector
                    bestMotionVector = motionVector

        return bestMotionVector
    
    def _getBlock(self, frame, motionVector, height, width):
        """
        Get block from frame.

        :param frame: Frame.
        :param motionVector: Motion vector.
        :param height: Height of block.
        :param width: Width of block.
        :return: Block.
        """

        # initialize block
        block = np.zeros((height, width), dtype=np.uint8)

        # iterate through block
        for y in range(height):
            for x in range(width):
                # get pixel
                pixel = self._getPixel(frame, motionVector, y, x)

                # set pixel
                block[y, x] = pixel

        return block
    
    def _getPixel(self, frame, motionVector, y, x):
        """
        Get pixel from frame.

        :param frame: Frame.
        :param motionVector: Motion vector.
        :param y: Y-coordinate of pixel.
        :param x: X-coordinate of pixel.
        :return: Pixel.
        """

        # get frame dimensions
        height, width = frame.shape

        # get pixel coordinates
        y = y + motionVector[1]
        x = x + motionVector[0]

        # limit pixel coordinates
        y = int(np.clip(y, 0, height - 1))
        x = int(np.clip(x, 0, width - 1))

        # return pixel
        return frame[y, x]