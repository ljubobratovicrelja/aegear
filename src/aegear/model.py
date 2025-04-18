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
        return self.forward_with_decoded(x)[0]
    
    def forward_with_decoded(self, x):
        """Forward pass with last decoder output return for interface with trajectory prediction."""
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

        return self.out(d0), d0


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        in_ch = input_channels + hidden_channels
        self.reset_gate = nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=padding)
        self.out_gate   = nn.Conv2d(in_ch, hidden_channels, kernel_size, padding=padding)

    def forward(self, x, h):
        if h is None:
            B, _, H, W = x.shape
            h = torch.zeros((B, self.hidden_channels, H, W), dtype=x.dtype, device=x.device)

        combined = torch.cat([x, h], dim=1)  # [B, input+hidden, H, W]
        z = torch.sigmoid(self.update_gate(combined))
        r = torch.sigmoid(self.reset_gate(combined))
        combined_reset = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.out_gate(combined_reset))
        h_new = (1 - z) * h + z * h_tilde
        return h_new


class ConvGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_steps):
        super().__init__()
        self.cell = ConvGRUCell(input_channels, hidden_channels)
        self.num_steps = num_steps

    def forward(self, input_seq):
        # input_seq: [B, T, C, H, W]
        B, T, C, H, W = input_seq.shape
        h = None
        for t in range(T):
            h = self.cell(input_seq[:, t], h)
        return h  # [B, hidden_channels, H, W]


class TemporalRefinedUNet(nn.Module):
    HISTORY_LEN = 3  # Used to define expected sequence length (not strictly needed here)

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet
        self.convgru = ConvGRU(input_channels=2, hidden_channels=8, num_steps=self.HISTORY_LEN)
        self.fusion = nn.Sequential(
            nn.Conv2d(8 + 8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1)
        )
    
    def forward(self, rgb, timestamp, history=None):
        raw_heatmap, decoder_feat = self.unet.forward_with_decoded(rgb)

        if history is None:
            return raw_heatmap

        heatmaps, timestamps = history
        B, T, _, H, W = heatmaps.shape

        dt = (timestamp[:, None] - timestamps).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        dt_map = dt.expand(-1, -1, 1, H, W)

        heat_seq = torch.cat([heatmaps, dt_map], dim=2)

        temporal_feat = self.convgru(heat_seq)
        fused = self.fusion(torch.cat([temporal_feat, decoder_feat], dim=1))

        return fused



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