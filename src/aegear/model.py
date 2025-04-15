import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import torch.nn.functional as F


class EfficientUNet(nn.Module):
    """
    Efficient U-Net model for image segmentation tasks.
    
    This model is based on the EfficientNet backbone and includes an encoder-decoder architecture, with skip connections
    between the encoder and decoder layers. The encoder downsamples the input image through a series of convolutional
    layers, while the decoder upsamples the feature maps back to the original image size.

    The model is designed to aid the tracking pipeline by providing a heatmap of the fish's location in the image, 
    at the same time serving as the classification of whether the fish is present or not, and its location in the image.

    The model is designed for training using the binary cross-entropy loss, hence the final output is expected to be
    run through a sigmoid activation function to produce a probability map.
    """
    def __init__(self, weights='IMAGENET1K_V1'):
        super().__init__()
        backbone = efficientnet_b0(weights=weights)
        features = list(backbone.features.children())

        # Encoder blocks with proper shape control
        self.enc1 = nn.Sequential(*features[:2])    # [B, 16, 112, 112]
        self.enc2 = nn.Sequential(*features[2:3])   # [B, 24, 56, 56]
        self.enc3 = nn.Sequential(*features[3:4])   # [B, 40, 28, 28]
        self.enc4 = nn.Sequential(*features[4:6])   # [B, 112, 14, 14]
        self.enc5 = nn.Sequential(*features[6:])    # [B, 1280, 7, 7]

        # Decoder blocks (in_channels = encoder out, out_channels = skip connection in)
        self.up4 = self._up_block(1280, 112)  # 7 → 14
        self.up3 = self._up_block(112, 40)    # 14 → 28
        self.up2 = self._up_block(40, 24)     # 28 → 56
        self.up1 = self._up_block(24, 16)     # 56 → 112
        self.up0 = self._up_block(16, 8)      # 112 → 224

        # Final conv to produce 1-channel heatmap
        self.out = nn.Conv2d(8, 1, kernel_size=1)

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.enc1(x)  # [B, 16, 112, 112]
        x2 = self.enc2(x1) # [B, 24, 56, 56]
        x3 = self.enc3(x2) # [B, 40, 28, 28]
        x4 = self.enc4(x3) # [B, 112, 14, 14]
        x5 = self.enc5(x4) # [B, 1280, 7, 7]

        d4 = self.up4(x5) + x4  # [B, 112, 14, 14]
        d3 = self.up3(d4) + x3  # [B, 40, 28, 28]
        d2 = self.up2(d3) + x2  # [B, 24, 56, 56]
        d1 = self.up1(d2) + x1  # [B, 16, 112, 112]
        d0 = self.up0(d1)       # [B, 8, 224, 224]

        return self.out(d0)   # [B, 1, 224, 224]


class TrajectoryPredictionNet(nn.Module):
    """
    GRU-based model for predicting short-term fish movement as a direction and intensity vector.

    Each input timestep consists of:
        - a pooled heatmap of the fish region
        - the relative position (x, y) of the fish compared to the current frame
        - the time difference (dt) from the current frame (in seconds)

    The model encodes the temporal sequence using a GRU and predicts:
        - a 2D unit direction vector (dx, dy)
        - a scalar intensity (speed in normalized coordinate units per second)
    """
    def __init__(self, heatmap_size=(224, 224), pooled_size=(64, 64),
                 hidden_dim=64, gru_layers=1):
        super().__init__()
        self.H, self.W = heatmap_size
        self.pooled_H, self.pooled_W = pooled_size
        self.input_dim = self.pooled_H * self.pooled_W + 2 + 1  # heatmap + rel pos + dt

        self.pool = nn.AdaptiveAvgPool2d(pooled_size)

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)  # dx, dy, intensity
        )

    def forward(self, heatmap_seq, rel_pos_seq, dt_seq):
        """
        Args:
            heatmap_seq (Tensor): Shape (B, T, 1, H, W), U-Net heatmaps
            rel_pos_seq (Tensor): Shape (B, T, 2), position deltas w.r.t. current frame
            dt_seq (Tensor): Shape (B, T, 1), time deltas w.r.t. current frame (in seconds)

        Returns:
            Tensor: Shape (B, 3), [direction_x, direction_y, intensity]
        """
        B, T, C, H, W = heatmap_seq.shape
        assert C == 1

        x = heatmap_seq.view(B * T, 1, H, W)
        pooled = self.pool(x).view(B, T, -1)  # (B, T, pooled_H * pooled_W)

        x = torch.cat([pooled, rel_pos_seq, dt_seq], dim=-1)  # (B, T, input_dim)

        _, h_n = self.gru(x)  # (num_layers, B, hidden_dim)
        h_last = h_n[-1]     # (B, hidden_dim)

        out = self.fc(h_last)  # (B, 3)
        return out.view(B, 3)


class ConvClassifier(nn.Module):
    """
    A simple convolutional network for binary classification.
    This model is designed to classify whether a fish is present in a given
    region of interest (ROI) of the image.
    """
    # Size of the region of interest (ROI) for classification.
    ROI_SIZE = 64

    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * (ConvClassifier.ROI_SIZE // 8) ** 2, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten layer
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))