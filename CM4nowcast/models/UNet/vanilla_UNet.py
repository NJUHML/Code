import torch
from torch import nn
import numpy as np


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):

    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False)
        )
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class DWCONV(nn.Module):
    """
    Depthwise Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=None):
        super(DWCONV, self).__init__()
        if groups == None:
            groups = in_channels
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups, bias=True
        )

    def forward(self, x):
        result = self.depthwise(x)
        return result


class UEncoder(nn.Module):

    def __init__(self, input_channel=6):
        super(UEncoder, self).__init__()
        self.res1 = DoubleConv(input_channel, 64)
        self.pool1 = nn.MaxPool2d(3)
        self.res2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(4)
        self.res3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.res4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.res5 = DoubleConv(512, 1024)
        self.pool5 = nn.MaxPool2d(2)

    def forward(self, x):
        features = []
        x = self.res1(x)
        features.append(x)  # (224, 224, 64)
        x = self.pool1(x)   # (112, 112, 64)

        x = self.res2(x)
        features.append(x)  # (112, 112, 128)
        x = self.pool2(x)   # (56, 56, 128)

        x = self.res3(x)
        features.append(x)  # (56, 56, 256)
        x = self.pool3(x)   # (28, 28, 256)

        x = self.res4(x)
        features.append(x)  # (28, 28, 512)
        x = self.pool4(x)   # (14, 14, 512)

        x = self.res5(x)
        features.append(x)  # (14, 14, 1024)
        x = self.pool5(x)  # (7, 7, 1024)
        features.append(x)
        return features


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            scale_factor = 2,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class UNet(nn.Module):
    def __init__(self, input_channel = 6, n_classes = 12,size = 384):
        super(UNet, self).__init__()
        self.p_encoder = UEncoder(input_channel=input_channel)
        self.encoder_channels = [1024, 512, 256, 128, 64]
        self.decoder1 =DecoderBlock(self.encoder_channels[0]+self.encoder_channels[0], self.encoder_channels[1])
        self.decoder2 =DecoderBlock(self.encoder_channels[1]+self.encoder_channels[1], self.encoder_channels[2])
        self.decoder3 =DecoderBlock(self.encoder_channels[2]+self.encoder_channels[2], self.encoder_channels[3])
        self.decoder4 = DecoderBlock(self.encoder_channels[3]+self.encoder_channels[3], self.encoder_channels[4], scale_factor = 4)
        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=n_classes,
            kernel_size=3,
        )
        self.decoder_final = DecoderBlock(in_channels=64, out_channels=64, scale_factor = 3)

    def forward(self, x):
        encoder_skips = self.p_encoder(x)

        x1_up = self.decoder1(encoder_skips[-1], encoder_skips[-2])
        x2_up = self.decoder2(x1_up, encoder_skips[-3])
        x3_up = self.decoder3(x2_up, encoder_skips[-4])
        x4_up = self.decoder4(x3_up, encoder_skips[-5])
        x_final = self.decoder_final(x4_up, None)
        logits = self.segmentation_head(x_final)
        return logits

