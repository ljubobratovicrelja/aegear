import torch
import torch.nn as nn
import torch.nn.functional as F


# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FishNet(nn.Module):
    def __init__(self, input_size=64, dropout_rate=0.5):
        super(FishNet, self).__init__()
        self.input_size = input_size

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.gn1 = nn.GroupNorm(8, 32)  # 32 channels, 8 groups
        self.pool = nn.MaxPool2d(2)

        # Block 2: 32 -> 64
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.gn2a = nn.GroupNorm(8, 64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.gn2b = nn.GroupNorm(8, 64)
        self.res_conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.se2 = SEBlock(64, reduction=16)

        # Block 3: 64 -> 128
        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.gn3a = nn.GroupNorm(8, 128)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.gn3b = nn.GroupNorm(8, 128)
        self.res_conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.se3 = SEBlock(128, reduction=16)

        # Block 4: 128 -> 256 (using a slightly larger kernel)
        self.conv4a = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.gn4a = nn.GroupNorm(8, 256)
        self.conv4b = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2)
        self.gn4b = nn.GroupNorm(8, 256)
        self.res_conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1)
        self.se4 = SEBlock(256, reduction=16)

        # Block 5: 256 -> 512
        self.conv5a = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.gn5a = nn.GroupNorm(8, 512)
        self.conv5b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.gn5b = nn.GroupNorm(8, 512)
        self.res_conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=1)
        self.se5 = SEBlock(512, reduction=16)

        # Additional channel attention
        self.attention = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 512, kernel_size=1),
            nn.Sigmoid()
        )

        # Global pooling and FC layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Block 1
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.pool(x)

        # Block 2 with residual connection + SE
        residual = self.res_conv2(x)
        x = F.relu(self.gn2a(self.conv2a(x)))
        x = self.gn2b(self.conv2b(x))
        x = F.relu(x + residual)
        x = self.se2(x)
        x = self.pool(x)

        # Block 3 with residual connection + SE
        residual = self.res_conv3(x)
        x = F.relu(self.gn3a(self.conv3a(x)))
        x = self.gn3b(self.conv3b(x))
        x = F.relu(x + residual)
        x = self.se3(x)
        x = self.pool(x)

        # Block 4 with residual connection + SE
        residual = self.res_conv4(x)
        x = F.relu(self.gn4a(self.conv4a(x)))
        x = self.gn4b(self.conv4b(x))
        x = F.relu(x + residual)
        x = self.se4(x)
        x = self.pool(x)

        # Block 5 with residual connection + SE
        residual = self.res_conv5(x)
        x = F.relu(self.gn5a(self.conv5a(x)))
        x = self.gn5b(self.conv5b(x))
        x = F.relu(x + residual)
        x = self.se5(x)
        x = self.pool(x)

        # Additional attention
        attn = self.attention(x)
        x = x * attn

        # Global Average Pooling and classifier
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


class ShallowFishNet(nn.Module):
    """
    ShallowFishNet: A lightweight CNN for fish vs. no-fish classification.

    Architecture Highlights:
    - 2 convolutional blocks, each with residual connections.
    - Narrower channel widths (16, 32) to reduce capacity.
    - Simple channel attention to focus on fish features.
    - High dropout in the fully connected layers to prevent overfitting.
    """

    def __init__(self, input_size=64, dropout_rate=0.5):
        super(ShallowFishNet, self).__init__()
        self.input_size = input_size

        # --- Block 1 (3 -> 16) ---
        self.conv1a = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.bn1a = nn.BatchNorm2d(16)
        self.conv1b = nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1)
        self.bn1b = nn.BatchNorm2d(16)
        self.res_conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1)

        # --- Block 2 (16 -> 32) ---
        self.conv2a = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.bn2a = nn.BatchNorm2d(32)
        self.conv2b = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.bn2b = nn.BatchNorm2d(32)
        self.res_conv2 = nn.Conv2d(16, 32, kernel_size=1, stride=1)

        # --- Attention (Channel-wise) ---
        self.attention = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.Sigmoid()
        )

        # --- Global Pooling & FC Layers ---
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # --- Block 1 ---
        residual = self.res_conv1(x)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = self.bn1b(self.conv1b(x))
        x = F.relu(x + residual)
        x = self.pool(x)

        # --- Block 2 ---
        residual = self.res_conv2(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = F.relu(x + residual)
        x = self.pool(x)

        # --- Attention ---
        attn = self.attention(x)
        x = x * attn

        # --- Classifier ---
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x
