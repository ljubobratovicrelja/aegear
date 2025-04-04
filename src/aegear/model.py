import torch.nn as nn
from torchvision.models import efficientnet_b0


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
    def __init__(self):
        super().__init__()
        backbone = efficientnet_b0(weights='IMAGENET1K_V1')
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
