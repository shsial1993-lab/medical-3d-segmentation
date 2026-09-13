from __future__ import annotations

import torch

from .model import Tiny3DUNet, binary_iou, dice_score


def synthetic_volume(size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-1, 1, size)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing='ij')
    mask = ((xx**2 + yy**2 + zz**2) < 0.35**2).float()
    volume = mask.mul(0.8).add(torch.rand_like(mask).mul(0.2))
    return volume[None, None], mask[None, None]


def main() -> None:
    torch.manual_seed(7)
    volume, target = synthetic_volume()
    model = Tiny3DUNet()
    with torch.no_grad():
        logits = model(volume)
    print('volume:', tuple(volume.shape), 'logits:', tuple(logits.shape))
    print('dice:', float(dice_score(logits, target)))
    print('iou:', float(binary_iou(logits, target)))


if __name__ == '__main__':
    main()
