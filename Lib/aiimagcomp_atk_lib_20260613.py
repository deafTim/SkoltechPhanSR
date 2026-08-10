"""
Local compression helpers for ADMM SR+compression notebooks (2026-06-13).
Replaces the lab aiimagcomp_atk_lib_3012 module; not a drop-in from the original repo.
Based on SWIN_train.py and process_div2k_bpp_selection_Colab.ipynb.
"""

from __future__ import annotations

import copy
import math
import os
import struct
from typing import Any, Dict, Union

import numpy as np
import torch

__all__ = [
    "copy",
    "bpp_loss",
    "ssim",
    "savecompressed",
    "save_checkpoint",
]


def bpp_loss(compression_output: Dict[str, Any], num_pixels: int) -> torch.Tensor:
    """Differentiable BPP estimate from CompressAI forward output."""
    ll_y = compression_output["likelihoods"]["y"]
    ll_z = compression_output["likelihoods"].get("z")
    bits = torch.log(ll_y).sum()
    if ll_z is not None:
        bits = bits + torch.log(ll_z).sum()
    return bits / (-math.log(2) * num_pixels)


def ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """SSIM for BCHW tensors in [0, 1]. Returns scalar tensor."""
    from skimage.metrics import structural_similarity as ssim_skimage

    a = img1.squeeze().detach().cpu().numpy()
    b = img2.squeeze().detach().cpu().numpy()

    if a.ndim == 3:
        a = np.transpose(a, (1, 2, 0))
        b = np.transpose(b, (1, 2, 0))
        a = np.clip(a, 0.0, 1.0)
        b = np.clip(b, 0.0, 1.0)
        val = ssim_skimage(a, b, channel_axis=2, data_range=1.0)
    else:
        a = np.clip(a, 0.0, 1.0)
        b = np.clip(b, 0.0, 1.0)
        val = ssim_skimage(a, b, data_range=1.0)

    return torch.tensor(val, dtype=img1.dtype, device=img1.device)


def _write_uints(fd, values) -> None:
    fd.write(struct.pack(">" + "I" * len(values), *values))


def _write_bytes(fd, data) -> None:
    if isinstance(data, (list, tuple)):
        data = data[0]
    fd.write(struct.pack(">I", len(data)))
    fd.write(data)


def savecompressed(
    compressfile: Union[str, os.PathLike],
    sr_output: Dict[str, Any],
    bitdepth: int,
    h: int,
    w: int,
) -> float:
    """
    Write CompressAI bitstream to disk and return BPP from file size (bits per pixel).
    bitdepth is kept for API compatibility with the original lib.
    """
    del bitdepth
    path = str(compressfile) + ".bin"
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "wb") as f:
        _write_uints(f, (h, w))
        for s in sr_output["strings"]:
            _write_bytes(f, s)

    num_pixels = h * w
    return (os.path.getsize(path) * 8) / num_pixels


def save_checkpoint(
    state: Dict[str, Any],
    path: Union[str, os.PathLike] = "checkpoint_interrupt.pth",
) -> None:
    """Save interrupt checkpoint (notebook passes only state dict)."""
    torch.save(state, str(path))
    print(f"Checkpoint saved: {path}")
