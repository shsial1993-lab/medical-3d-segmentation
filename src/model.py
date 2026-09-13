from __future__ import annotations

import torch
from torch import nn


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.block(volume)


class Tiny3DUNet(nn.Module):
    """A small volumetric segmentation model for baseline experiments."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.enc1 = DoubleConv3D(in_channels, 8)
        self.enc2 = DoubleConv3D(8, 16)
        self.bottleneck = DoubleConv3D(16, 32)
        self.pool = nn.MaxPool3d(2)
        self.up2 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec2 = DoubleConv3D(32, 16)
        self.up1 = nn.ConvTranspose3d(16, 8, 2, stride=2)
        self.dec1 = DoubleConv3D(16, 8)
        self.head = nn.Conv3d(8, out_channels, 1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(volume)
        skip2 = self.enc2(self.pool(skip1))
        bottleneck = self.bottleneck(self.pool(skip2))
        decoded2 = self.up2(bottleneck)
        decoded2 = self.dec2(torch.cat((decoded2, skip2), dim=1))
        decoded1 = self.up1(decoded2)
        decoded1 = self.dec1(torch.cat((decoded1, skip1), dim=1))
        return self.head(decoded1)


def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prediction = (logits.sigmoid() > 0.5).float()
    target = target.float()
    intersection = (prediction * target).sum()
    return (2 * intersection + eps) / (prediction.sum() + target.sum() + eps)


def binary_iou(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prediction = (logits.sigmoid() > 0.5).float()
    target = target.float()
    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)
