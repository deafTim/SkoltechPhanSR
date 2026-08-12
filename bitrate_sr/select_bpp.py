#!/usr/bin/env python3
"""Rank DIV2K images by cheng BPP after pretrained SR; write diverse IMG_LIST.

Prefer: edit scripts/bitrate_sr_config.sh, then bash scripts/select_bpp_images.sh
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_kwargs):
        return it

from .data import DEFAULT_PROJECT_ROOT, build_compression, build_sr_pretrained
from .metrics import forward_sr


def center_crop_lr(path: Path, crop_size: int, scaled_size: int, to_t):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < crop_size or h < crop_size:
        return None
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    crop = img.crop((left, top, left + crop_size, top + crop_size))
    lr = crop.resize((scaled_size, scaled_size), Image.BICUBIC)
    return to_t(lr)


def bpp_per_item(likelihoods: dict, index: int, num_pixels: int) -> float:
    total = 0.0
    for lik in likelihoods.values():
        total += float(torch.log(lik[index]).sum().item())
    return total / (-math.log(2) * num_pixels)


def compute_bpp_all(
    image_files: list[Path],
    model_sr,
    model_compression,
    device,
    crop_size: int,
    scaled_size: int,
    batch_size: int,
) -> list[dict]:
    to_t = transforms.ToTensor()
    results: list[dict] = []
    for start in tqdm(range(0, len(image_files), batch_size), desc="BPP", unit="batch"):
        batch_paths = image_files[start : start + batch_size]
        tensors = []
        infos = []
        for p in batch_paths:
            try:
                t = center_crop_lr(p, crop_size, scaled_size, to_t)
            except Exception as e:
                print(f"skip {p.name}: {e}")
                continue
            if t is None:
                continue
            tensors.append(t)
            infos.append({"filename": p.name, "path": str(p)})
        if not tensors:
            continue
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            sr = forward_sr(model_sr, batch).clamp(0, 1)
            out = model_compression(sr)
            n_pix = sr.shape[2] * sr.shape[3]
            for i, info in enumerate(infos):
                bpp = bpp_per_item(out["likelihoods"], i, n_pix)
                results.append({**info, "bpp": bpp})
    return results


def select_diverse(sorted_rows: list[dict], num_selected: int) -> list[dict]:
    if not sorted_rows:
        return []
    n = min(num_selected, len(sorted_rows))
    if n == len(sorted_rows):
        return list(sorted_rows)
    group = len(sorted_rows) // n
    picked = []
    for i in range(n):
        idx = i * group + group // 2
        if idx < len(sorted_rows):
            picked.append(sorted_rows[idx])
    return picked


def build_argparser():
    p = argparse.ArgumentParser(description="Select diverse DIV2K images by BPP")
    p.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    p.add_argument("--data-dir", default="")
    p.add_argument("--img-list-out", required=True, help="Output txt for training IMG_LIST")
    p.add_argument("--bpp-csv", default="", help="Optional full sorted ranking CSV")
    p.add_argument("--num-selected", type=int, default=10)
    p.add_argument("--backbone", default="nina", choices=["nina", "swin"])
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--scaled-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--compression-model", default="cheng2020-attn")
    p.add_argument("--compression-quality", type=int, default=6)
    p.add_argument("--device", default="")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    project = Path(args.project_root)
    data_dir = Path(args.data_dir) if args.data_dir else project / "DIV2K" / "DIV2K_train_HR"
    if not data_dir.is_dir():
        print(f"DATA_DIR not found: {data_dir}", file=sys.stderr)
        return 1

    image_files = sorted(data_dir.glob("*.png"))
    if not image_files:
        print(f"No PNGs in {data_dir}", file=sys.stderr)
        return 1

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device} images={len(image_files)} backbone={args.backbone}")

    model_sr = build_sr_pretrained(args.backbone, device)
    model_compression = build_compression(device, args.compression_model, args.compression_quality)

    rows = compute_bpp_all(
        image_files,
        model_sr,
        model_compression,
        device,
        args.crop_size,
        args.scaled_size,
        args.batch_size,
    )
    if not rows:
        print("No BPP results", file=sys.stderr)
        return 1

    rows_sorted = sorted(rows, key=lambda r: r["bpp"])
    print(f"BPP range: {rows_sorted[0]['bpp']:.4f} .. {rows_sorted[-1]['bpp']:.4f}")

    selected = select_diverse(rows_sorted, args.num_selected)
    out_list = Path(args.img_list_out)
    out_list.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Diverse BPP selection (backbone={args.backbone}, n={len(selected)})",
        "# filename,bpp",
    ]
    for r in selected:
        lines.append(f"{r['filename']},{r['bpp']:.6f}")
        print(f"  {r['filename']}: BPP={r['bpp']:.4f}")
    out_list.write_text("\n".join(lines) + "\n")
    print(f"Wrote IMG_LIST -> {out_list}")

    if args.bpp_csv:
        csv_path = Path(args.bpp_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["filename", "bpp", "path"])
            w.writeheader()
            w.writerows(rows_sorted)
        print(f"Wrote full ranking -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
