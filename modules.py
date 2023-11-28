from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, is_last: bool = False):
        super().__init__()
        self.is_last = is_last
        self.conv = nn.Conv2d(in_channel, out_channel, stride=1, padding=1, kernel_size=3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if not self.is_last:
            out = self.relu(out)
        return out

class NoisePredictor(nn.Module):
    def __init__(self):
        super(NoisePredictor, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        return x

class up(nn.Module):
    def __init__(self, in_ch: int):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = x2 + x1
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.first = nn.Sequential(
            BasicConv(6, 64),
            BasicConv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            BasicConv(64, 128),
            BasicConv(128, 128),
            BasicConv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            BasicConv(128, 256),
            BasicConv(256, 256),
            BasicConv(256, 256),
            BasicConv(256, 256),
            BasicConv(256, 256),
            BasicConv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            BasicConv(128, 128),
            BasicConv(128, 128),
            BasicConv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            BasicConv(64, 64),
            BasicConv(64, 64)
        )

        self.outc = BasicConv(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.first(x)
        down1 = self.down1(first)
        conv1 = self.conv1(down1)
        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)
        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)
        up2 = self.up2(conv3, first)
        conv4 = self.conv4(up2)
        out = self.outc(conv4)
        return out
