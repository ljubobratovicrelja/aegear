"""
Classifier Model for Object Classification

This PyTorch model is designed for single-class object classification tasks.
It consists of 4 convolutional layers followed by max-pooling operations, 
and 3 fully connected layers. A dropout layer is added for regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    """
    Object Classification Model.

    Attributes:
        conv1 (nn.Module): First convolutional layer.
        conv2 (nn.Module): Second convolutional layer.
        conv3 (nn.Module): Third convolutional layer.
        conv4 (nn.Module): Fourth convolutional layer.
        fc1 (nn.Module): First fully connected layer.
        fc2 (nn.Module): Second fully connected layer.
        fc3 (nn.Module): Third fully connected layer.
        dropout (nn.Module): Dropout layer for regularization.
    """

    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Additional layer
        self.fc1 = nn.Linear(1024, 256)  # Increased size
        self.fc2 = nn.Linear(256, 64)    # New layer
        self.fc3 = nn.Linear(64, 1)      # Output layer
        
        self.dropout = nn.Dropout(0.5)  # Dropout layer for regularization

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Using dropout for regularization
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x