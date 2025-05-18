import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import torch.nn.functional as F


class CBAM(nn.Module):
    """Lightweight convolutional block attention module (CBAM) for channel and spatial attention."""

    def __init__(self, in_channels):
        super().__init__()
        # Channel attention
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel(x)
        x = x * ca

        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial(torch.cat([max_pool, avg_pool], dim=1))
        return x * sa


class EfficientUNet(nn.Module):
    """
    EfficientUNet backbone based on EfficientNet-B0, enhanced with CBAM
    (Convolutional Block Attention Module) attention blocks after each encoder
    and decoder stage.

    The architecture removes the deepest (last) encoder and
    decoder stages compared to a standard UNet, resulting in a lighter model
    with fewer parameters and reduced memory usage, while retaining strong
    feature extraction and localization capabilities.

    CBAM modules are used to improve feature representation by applying both
    channel and spatial attention at multiple levels of the network, allowing
    the model to focus on the object of interest while ignoring irrelevant information.
    This is particularly useful in scenarios where the object of interest (e.g., fish)
    may be small and difficult to distinguish from the background, or when there
    are multiple objects present in the image.
    """

    def __init__(self, weights=None):
        super().__init__()
        backbone = efficientnet_b0(weights=weights)
        features = list(backbone.features.children())

        # Encoder stages
        self.enc1 = nn.Sequential(*features[:2])  # Output: 16 ch, S/2
        self.enc2 = nn.Sequential(*features[2:3])  # Output: 24 ch, S/4
        self.enc3 = nn.Sequential(*features[3:4])  # Output: 40 ch, S/8
        self.enc4 = nn.Sequential(*features[4:5])  # Output: 80 ch, S/16
        self.enc5 = nn.Sequential(*features[5:6])  # Output: 112 ch, S/16

        # Bottleneck with dilated convs.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(112, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.att_bottleneck = CBAM(256)

        # Decoder with CBAM after skip merges
        self.att4 = CBAM(256 + 112)
        self.up4 = self._conf_block(256 + 112, 64)  # S/16 -> S/16

        self.att3 = CBAM(64 + 80)
        self.up3 = self._up_block(64 + 80, 32)

        self.att2 = CBAM(32 + 40)
        self.up2 = self._up_block(32 + 40, 24)

        self.att1 = CBAM(24 + 24)
        self.up1 = self._up_block(24 + 24, 16)

        self.att0 = CBAM(16 + 16)
        self.up0 = self._up_block(16 + 16, 8)

        # Final 1-channel output
        self.out = nn.Conv2d(8, 1, kernel_size=1)

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _conf_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.forward_with_decoded(x)[0]

    def forward_with_decoded(self, x):
        # Encoder
        x1 = self.enc1(x)  # S/2
        x2 = self.enc2(x1)  # S/4
        x3 = self.enc3(x2)  # S/8
        x4 = self.enc4(x3)  # S/16
        x5 = self.enc5(x4)  # S/16

        b = self.bottleneck(x5)
        b = self.att_bottleneck(b)

        # Decoder
        d4_cat = torch.cat([b, x5], dim=1)
        d4_att = self.att4(d4_cat)
        d4 = self.up4(d4_att)

        d3_cat = torch.cat([d4, x4], dim=1)
        d3_att = self.att3(d3_cat)
        d3 = self.up3(d3_att)

        d2_cat = torch.cat([d3, x3], dim=1)
        d2_att = self.att2(d2_cat)
        d2 = self.up2(d2_att)

        d1_cat = torch.cat([d2, x2], dim=1)
        d1_att = self.att1(d1_cat)
        d1 = self.up1(d1_att)

        d0_cat = torch.cat([d1, x1], dim=1)
        d0_att = self.att0(d0_cat)
        d0 = self.up0(d0_att)

        # Final output
        out = self.out(d0)

        # Resize to original input size
        out = F.interpolate(out,
                            size=x.shape[2:],
                            mode='bilinear',
                            align_corners=False)

        return out, d0


class SiameseTracker(nn.Module):
    """
    Siamese UNet model for tracking, based on EfficientUNet.

    This model is designed to take two inputs: a template image and a search
    image. The template image is the reference image of the object to be
    tracked, while the search image is the current frame in which the object
    is being searched for. The model processes both images through a shared
    UNet architecture, extracting features from both images and then
    concatenating them at each stage of the decoder. This allows the model to
    leverage the spatial information from both images, improving the
    tracking performance.
    """

    def __init__(self, unet=EfficientUNet()):
        super().__init__()
        # Share encoder stages from the UNet
        self.enc1 = unet.enc1
        self.enc2 = unet.enc2
        self.enc3 = unet.enc3
        self.enc4 = unet.enc4
        self.enc5 = unet.enc5

        # Share bottleneck and bottleneck attention
        self.bottleneck = unet.bottleneck
        self.att_bottleneck = unet.att_bottleneck

        # Decoder blocks with adjusted input channel sizes for concatenated Siamese features
        # The input channels to att/up blocks will be double the UNet's combined input
        self.att4 = CBAM(256 * 2 + 112 * 2)
        self.up4 = unet._conf_block(256 * 2 + 112 * 2, 64)

        self.att3 = CBAM(64 + 80 * 2)
        self.up3 = unet._up_block(64 + 80 * 2, 32)

        self.att2 = CBAM(32 + 40 * 2)
        self.up2 = unet._up_block(32 + 40 * 2, 24)

        self.att1 = CBAM(24 + 24 * 2)
        self.up1 = unet._up_block(24 + 24 * 2, 16)

        self.att0 = CBAM(16 + 16 * 2)
        self.up0 = unet._up_block(16 + 16 * 2, 8)

        # Re-use the output layer from UNet
        self.out = unet.out

    def forward(self, template, search):
        # Encoder
        t1 = self.enc1(template)  # S/2
        s1 = self.enc1(search)

        t2 = self.enc2(t1)  # S/4
        s2 = self.enc2(s1)

        t3 = self.enc3(t2)  # S/8
        s3 = self.enc3(s2)

        t4 = self.enc4(t3)  # S/16
        s4 = self.enc4(s3)

        t5 = self.enc5(t4)  # S/16
        s5 = self.enc5(s4)

        # Bottleneck with attention.
        b_t = self.bottleneck(t5)
        b_s = self.bottleneck(s5)
        b_t_att = self.att_bottleneck(b_t)
        b_s_att = self.att_bottleneck(b_s)

        fused_bottleneck = torch.cat(
            [b_t_att, b_s_att], dim=1)

        # Decoder
        _d4_cat = torch.cat(
            [fused_bottleneck, torch.cat([t5, s5], dim=1)], dim=1)
        _d4_att = self.att4(_d4_cat)
        d4_fused = self.up4(_d4_att)

        _d3_cat = torch.cat([d4_fused, torch.cat([t4, s4], dim=1)], dim=1)
        _d3_att = self.att3(_d3_cat)
        d3_fused = self.up3(_d3_att)

        _d2_cat = torch.cat([d3_fused, torch.cat([t3, s3], dim=1)], dim=1)
        _d2_att = self.att2(_d2_cat)
        d2_fused = self.up2(_d2_att)

        _d1_cat = torch.cat([d2_fused, torch.cat([t2, s2], dim=1)], dim=1)
        _d1_att = self.att1(_d1_cat)
        d1_fused = self.up1(_d1_att)

        _d0_cat = torch.cat([d1_fused, torch.cat([t1, s1], dim=1)], dim=1)
        _d0_att = self.att0(_d0_cat)
        d0_fused = self.up0(_d0_att)

        out = self.out(d0_fused)
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
