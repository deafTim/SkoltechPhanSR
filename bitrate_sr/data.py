"""Data loading and pretrained model builders."""

from __future__ import annotations

import os
import ssl
import subprocess
from pathlib import Path
from shutil import which

import torch
from PIL import Image
from torchvision import transforms

from .lora import _nina_attn_expand_ok, inject_lora_conv2d, inject_lora_linear

# Zhores often has broken CA store for torch.hub downloads
ssl._create_default_https_context = ssl._create_unverified_context

DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("SCIENCE_PHAN_ROOT", "/gpfs/gpfs0/timofey.glukhikh/Science_Phan")
)

NINA_URL = "https://github.com/Coloquinte/torchSR/releases/download/v1.0.3/ninasr_b0_x2.pt"
CHENG_URL = "https://compressai.s3.amazonaws.com/models/v1/cheng2020_attn-mse-6-730501f2.pth.tar"
SWIN_MODEL_ID = "caidas/swin2SR-classical-sr-x2-64"


def ensure_hub_file(filename: str, url: str) -> Path:
    path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1000:
        print(f"Cached: {path}")
        return path
    print(f"Downloading {filename} ...")
    if which("wget"):
        subprocess.check_call(["wget", "--no-check-certificate", "-O", str(path), url])
    elif which("curl"):
        subprocess.check_call(["curl", "-k", "-L", "-o", str(path), url])
    else:
        import urllib.request

        urllib.request.urlretrieve(url, path)
    return path


def load_lr_hr_pair(
    img_path: Path,
    crop_size: int = 512,
    scaled_size: int = 256,
    left: int = 1328,
    top: int = 826,
    device: torch.device | None = None,
):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if w < crop_size or h < crop_size:
        raise ValueError(f"Image too small: {img_path} ({w}x{h})")
    left = min(max(0, left), w - crop_size)
    top = min(max(0, top), h - crop_size)
    crop = img.crop((left, top, left + crop_size, top + crop_size))
    lr = crop.resize((scaled_size, scaled_size), Image.BICUBIC)
    to_t = transforms.ToTensor()
    x = to_t(lr).unsqueeze(0).to(device)
    gt = to_t(crop).unsqueeze(0).to(device)
    meta = {"left": left, "top": top, "crop_size": crop_size, "scaled_size": scaled_size}
    return x, gt, meta


def _assert_only_lora_trainable(model: torch.nn.Module, label: str, n_layers: int):
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"{label}: LoRA inject failed — no trainable params")
    bad = [n for n, _ in trainable if "lora_" not in n]
    if bad:
        raise RuntimeError(f"{label}: unexpected trainable params: {bad[:10]}")
    print(
        f"{label}: layers={n_layers}, trainable={sum(p.numel() for _, p in trainable)}, "
        f"total={sum(p.numel() for p in model.parameters())}"
    )


def build_nina_lora(
    device,
    lora_r: int = 4,
    lora_alpha: float = 8.0,
    lora_target: str = "all",
):
    """lora_target: all | attention_expand (only body.*.body.2.body.3)."""
    from torchsr.models import ninasr_b0

    ensure_hub_file("ninasr_b0_x2.pt", NINA_URL)
    model = ninasr_b0(scale=2, pretrained=True).to(device)
    for p in model.parameters():
        p.requires_grad = False
    target = (lora_target or "all").lower()
    if target in ("all", ""):
        filt = None
    elif target == "attention_expand":
        filt = _nina_attn_expand_ok
    else:
        raise ValueError(f"Unknown lora_target={lora_target!r} (use all|attention_expand)")
    n = inject_lora_conv2d(model, r=lora_r, alpha=lora_alpha, name_filter=filt)
    model = model.to(device)
    _assert_only_lora_trainable(model, f"Nina+LoRA[{target}]", n)
    return model


def build_swin_lora(
    device,
    lora_r: int = 4,
    lora_alpha: float = 8.0,
    model_id: str = SWIN_MODEL_ID,
):
    """Swin2SR + LoRA on attention/MLP Linear layers (same ADMM as Nina)."""
    from transformers import Swin2SRForImageSuperResolution

    model = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(device)
    for p in model.parameters():
        p.requires_grad = False
    n = inject_lora_linear(model, r=lora_r, alpha=lora_alpha)
    if n == 0:
        # fallback: wrap all Linear if name filter matched nothing
        n = inject_lora_linear(model, r=lora_r, alpha=lora_alpha, name_filter=lambda _: True)
    model = model.to(device)
    _assert_only_lora_trainable(model, "Swin+LoRA", n)
    return model


def build_sr_lora(
    backbone: str,
    device,
    lora_r: int = 4,
    lora_alpha: float = 8.0,
    lora_target: str = "all",
):
    backbone = backbone.lower()
    if backbone == "nina":
        return build_nina_lora(device, lora_r, lora_alpha, lora_target=lora_target)
    if backbone == "swin":
        if (lora_target or "all").lower() not in ("all", ""):
            print(f"Warning: lora_target={lora_target!r} ignored for swin (Linear filter is fixed)")
        return build_swin_lora(device, lora_r, lora_alpha)
    raise ValueError(f"Unknown backbone: {backbone} (use nina|swin)")


def build_sr_pretrained(backbone: str, device):
    """Frozen pretrained SR (no LoRA) — for BPP ranking / baselines."""
    backbone = backbone.lower()
    if backbone == "nina":
        from torchsr.models import ninasr_b0

        ensure_hub_file("ninasr_b0_x2.pt", NINA_URL)
        model = ninasr_b0(scale=2, pretrained=True).to(device)
    elif backbone == "swin":
        from transformers import Swin2SRForImageSuperResolution

        model = Swin2SRForImageSuperResolution.from_pretrained(SWIN_MODEL_ID).to(device)
    else:
        raise ValueError(f"Unknown backbone: {backbone} (use nina|swin)")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def build_compression(device, model_name: str = "cheng2020-attn", quality: int = 6):
    """Frozen CompressAI model. Default eval() for metrics/select.

    ADMM switches to train() temporarily so rate is differentiable (noise vs quant).
    """
    from compressai.zoo import models

    ensure_hub_file("cheng2020_attn-mse-6-730501f2.pth.tar", CHENG_URL)
    model = models[model_name](quality=quality, pretrained=True).to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def parse_img_list(path: Path) -> list[str]:
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line.split(",")[0].strip())
    return names


_LORA_TARGET_TAG = {
    "attention_expand": "attnexp",
}


def run_tag(
    img_stem: str,
    target_psnr: float,
    lam: float,
    backbone: str,
    lora_r: int,
    method: str = "admm",
    lora_target: str = "all",
) -> str:
    psnr_s = str(int(target_psnr)) if float(target_psnr).is_integer() else str(target_psnr)
    base = f"img{img_stem}_{backbone}_psnr{psnr_s}_lam{lam:g}_r{lora_r}"
    extras: list[str] = []
    if method:
        extras.append(method)
    lt = (lora_target or "all").lower()
    if lt not in ("all", ""):
        extras.append(_LORA_TARGET_TAG.get(lt, lt.replace("_", "")))
    if extras:
        return f"{base}_{'_'.join(extras)}"
    return base
