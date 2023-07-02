"""
Created on Mon Apr  6 15:40:00 2020

This file contains the classifier model for the project. It is a simple CNN
with 3 convolutional layers and 2 fully connected layers. The output is a
single value between 0 and 1, where 0 means no motion and 1 means motion.

The model is trained on the dataset in the data folder. The dataset contains
images of size 32x32. The images are divided into two folders, one for motion
and one for no motion. The images are named as follows:
    - motion_0.png
    - motion_1.png
    - ...
    - motion_999.png
    - no_motion_0.png
    - no_motion_1.png
    - ...
    - no_motion_999.png

The model is trained on 80% of the images and tested on the remaining 20%.

The model is trained using the Adam optimizer and the binary cross entropy
loss function. The model is trained for 150 epochs with a batch size of 64.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x