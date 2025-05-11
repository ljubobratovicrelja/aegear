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
        self.enc5 = nn.Sequential(*features[6:])

        # Decoder blocks (in_channels = encoder out, out_channels = skip connection in)
        self.up4 = self._up_block(1280, 112)
        self.up3 = self._up_block(112, 40)
        self.up2 = self._up_block(40, 24)
        self.up1 = self._up_block(24, 16)
        self.up0 = self._up_block(16, 8)

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
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        d4 = self.up4(x5) + x4
        d3 = self.up3(d4) + x3
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

        # Reuse trained encoder from existing EfficientUNet instance
        self.encoder = nn.Sequential(
            unet.enc1,
            unet.enc2,
            unet.enc3,
        )

        HEAD_WIDTH = 256

        # Feature normalization layer
        self.normalize = nn.GroupNorm(num_groups=16, num_channels=HEAD_WIDTH)

        # Initial reduction layer
        self.initial = nn.Sequential(
            nn.Conv2d(80, HEAD_WIDTH, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.resblock1 = nn.Sequential(
            nn.Conv2d(HEAD_WIDTH, HEAD_WIDTH, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(HEAD_WIDTH, HEAD_WIDTH, kernel_size=3, padding=1)
        )

        self.resblock2 = nn.Sequential(
            nn.Conv2d(HEAD_WIDTH, HEAD_WIDTH, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(HEAD_WIDTH, HEAD_WIDTH, kernel_size=3, padding=1)
        )

        self.resblock3 = nn.Sequential(  # New extra block
            nn.Conv2d(HEAD_WIDTH, HEAD_WIDTH, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(HEAD_WIDTH, HEAD_WIDTH, kernel_size=3, padding=1)
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(HEAD_WIDTH, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, template, search):
        feat_t = self.encoder(template)
        feat_s = self.encoder(search)

        feat_t = F.normalize(feat_t, p=2, dim=1)
        feat_s = F.normalize(feat_s, p=2, dim=1)

        x = torch.cat([feat_t, feat_s], dim=1)

        x = self.initial(x)
        x = self.normalize(x)

        res = self.resblock1(x)
        x = x + res

        res = self.resblock2(x)
        x = x + res

        res = self.resblock3(x)
        x = x + res

        return self.upsample(x)



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