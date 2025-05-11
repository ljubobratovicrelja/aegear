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
        self.enc1 = nn.Sequential(*features[:2])
        self.enc2 = nn.Sequential(*features[2:3])
        self.enc3 = nn.Sequential(*features[3:4])
        self.enc4 = nn.Sequential(*features[4:6])

        # Decoder blocks (in_channels = encoder out, out_channels = skip connection in)
        self.up3 = self._up_block(112, 40)
        self.up2 = self._up_block(40, 24)
        self.up1 = self._up_block(24, 16)
        self.up0 = self._up_block(16, 8)

        # Final conv to produce 1-channel heatmap
        self.out = nn.Conv2d(8, 1, kernel_size=1)

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        return self.forward_with_decoded(x)[0]
    
    def forward_with_decoded(self, x):
        """Forward pass with last decoder output return for interface with trajectory prediction."""
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        d3 = self.up3(x4) + x3
        d2 = self.up2(d3) + x2
        d1 = self.up1(d2) + x1
        d0 = self.up0(d1)

        out = self.out(d0)

        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out, d0


class SiameseTracker(nn.Module):
    """
    Siamese tracker using EfficientNet-B0 encoder features shared with a pretrained EfficientUNet.
    Outputs a high-resolution response map using residual blocks and upsampling.
    """
    def __init__(self, unet=EfficientUNet()):
        super().__init__()
        
        # Share encoders
        self.enc1 = unet.enc1
        self.enc2 = unet.enc2
        self.enc3 = unet.enc3
        self.enc4 = unet.enc4

        # Decoder with adjusted input channel sizes (concatenated feature maps)
        self.up3 = unet._up_block(112 * 2, 80)
        self.up2 = unet._up_block(80, 48)
        self.up1 = unet._up_block(48, 32)
        self.up0 = unet._up_block(32, 8)
        self.out = unet.out  # Also use the output from the EfficientUNet.

    def forward(self, template, search):
        # Encode both
        t1, t2, t3, t4 = self.enc1(template), None, None, None
        s1, s2, s3, s4 = self.enc1(search), None, None, None

        t2 = self.enc2(t1)
        s2 = self.enc2(s1)

        t3 = self.enc3(t2)
        s3 = self.enc3(s2)

        t4 = self.enc4(t3)
        s4 = self.enc4(s3)

        # Fuse at each level (concat)
        f4 = torch.cat([t4, s4], dim=1)
        f3 = torch.cat([t3, s3], dim=1)
        f2 = torch.cat([t2, s2], dim=1)
        f1 = torch.cat([t1, s1], dim=1)

        d3 = self.up3(f4) + f3
        d2 = self.up2(d3) + f2
        d1 = self.up1(d2) + f1
        d0 = self.up0(d1)

        out = self.out(d0)
        return F.interpolate(out, size=template.shape[2:], mode='bilinear', align_corners=False)


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