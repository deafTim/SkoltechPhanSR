"""Metrics and small tensor helpers."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def bpp_loss(output, num_pixels: int) -> torch.Tensor:
    return sum(
        torch.log(lik).sum() / (-math.log(2) * num_pixels)
        for lik in output["likelihoods"].values()
    )


def psnr(a: torch.Tensor, b: torch.Tensor, max_i: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(a, b)
    return 10 * torch.log10((max_i ** 2) / (mse + 1e-12))


def ssim_np(a: torch.Tensor, b: torch.Tensor) -> float | None:
    try:
        from skimage.metrics import structural_similarity as ssim_sk
    except ImportError:
        return None
    x = a.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    y = b.detach().squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return float(ssim_sk(x, y, channel_axis=2, data_range=1.0))


def approx_file_bpp(model_compression, image: torch.Tensor) -> float | None:
    try:
        h, w = image.shape[-2:]
        with torch.no_grad():
            out = model_compression.compress(image)
        nbytes = sum(len(s) for group in out["strings"] for s in group)
        return float(nbytes) * 8.0 / float(h * w)
    except Exception as e:
        print(f"Warning: file bpp failed ({e})")
        return None


def eval_perf_compression(model_compression, sr: torch.Tensor, ref: torch.Tensor | None = None):
    with torch.no_grad():
        out = model_compression(sr)
        x_hat = out["x_hat"].clamp(0, 1)
        n_pix = sr.shape[2] * sr.shape[3]
        result = {
            "PSNR": float(psnr(x_hat, sr).cpu()),
            "Bpp": float(bpp_loss(out, n_pix).cpu()),
            "Bpp(fsize)": approx_file_bpp(model_compression, sr),
            "SSIM": ssim_np(x_hat, sr),
            "PSNR_cmpref": None,
            "SSI_cmpref": None,
        }
        if ref is not None:
            result["PSNR_cmpref"] = float(psnr(x_hat, ref).cpu())
            result["SSI_cmpref"] = ssim_np(x_hat, ref)
    return result


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.detach().float().cpu().clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def forward_sr(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if hasattr(out, "reconstruction") and out.reconstruction is not None:
        return out.reconstruction
    if isinstance(out, dict) and "reconstruction" in out:
        return out["reconstruction"]
    if torch.is_tensor(out):
        return out
    raise TypeError(f"Unexpected SR output type: {type(out)}")
