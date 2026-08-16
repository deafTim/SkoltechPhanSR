#!/usr/bin/env python3
"""Bitrate-minimizing SR with LoRA (NinaSR or Swin2SR).

Methods: ADMM (default) or direct one-loss Adam.
Prefer: edit scripts/bitrate_sr_config.sh, then bash scripts/run_sweep.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .admm import minbitrate_superresolution
from .data import (
    DEFAULT_PROJECT_ROOT,
    build_compression,
    build_sr_lora,
    load_lr_hr_pair,
    parse_img_list,
    run_tag,
)
from .direct import minbitrate_superresolution_direct
from .metrics import eval_perf_compression, forward_sr, tensor_to_pil


def run_one(args, img_name: str, model_compression) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    project = Path(args.project_root)
    data_dir = Path(args.data_dir) if args.data_dir else project / "DIV2K" / "DIV2K_train_HR"
    runs_dir = Path(args.runs_dir) if args.runs_dir else project / "runs" / "bitrate_sr"

    img_path = data_dir / img_name
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    method = (args.method or "admm").lower()
    lora_target = (getattr(args, "lora_target", None) or "all").lower()
    tag = run_tag(
        Path(img_name).stem,
        args.target_psnr,
        args.lam,
        args.backbone,
        args.lora_r,
        method=method,
        lora_target=lora_target,
    )
    out_dir = runs_dir / tag
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file():
        print(f"=== skip {tag} (exists {metrics_path}) ===")
        return json.loads(metrics_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {tag} ===\nmethod={method} lora_target={lora_target} device={device} out_dir={out_dir}")

    x, gt, crop_meta = load_lr_hr_pair(
        img_path,
        crop_size=args.crop_size,
        scaled_size=args.scaled_size,
        left=args.crop_left,
        top=args.crop_top,
        device=device,
    )

    model = build_sr_lora(
        args.backbone,
        device,
        args.lora_r,
        args.lora_alpha,
        lora_target=lora_target,
    )
    with torch.no_grad():
        sr_orig = forward_sr(model, x).detach()

    Z = T = None
    if method == "admm":
        model, Z, T, loss_h, bpp_h, psnr_h = minbitrate_superresolution(
            x,
            sr_orig,
            args.target_psnr,
            args.lam,
            args.inners,
            model,
            model_compression,
            lr=args.lr,
            outer_iterations=args.outers,
            checkpoint_dir=out_dir,
        )
    elif method == "direct":
        model, _sr_last, loss_h, bpp_h, psnr_h = minbitrate_superresolution_direct(
            x,
            gt,
            sr_orig,
            args.target_psnr,
            args.lam,
            model,
            model_compression,
            lr=args.lr,
            steps=args.steps,
            checkpoint_dir=out_dir,
            checkpoint_every=args.checkpoint_every,
            ref=args.direct_ref,
        )
    else:
        raise ValueError(f"Unknown method={method!r} (use admm|direct)")

    model.eval()
    with torch.no_grad():
        sr_final = forward_sr(model, x).detach()
    metrics = eval_perf_compression(model_compression, sr_final, gt)
    metrics.update(
        {
            "img": img_name,
            "backbone": args.backbone,
            "method": method,
            "lora_target": lora_target,
            "lambda": args.lam,
            "target_psnr": args.target_psnr,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lr": args.lr,
            "inners": args.inners,
            "outers": args.outers,
            "steps": args.steps if method == "direct" else args.outers * args.inners,
            "direct_ref": args.direct_ref if method == "direct" else None,
            "crop": crop_meta,
            "tag": tag,
            "bpp_train_end": float(bpp_h[-1]) if bpp_h else None,
            "bpp_train_min": float(min(bpp_h)) if bpp_h else None,
        }
    )
    if Z is not None:
        metrics_z = eval_perf_compression(model_compression, Z, gt)
        metrics["Bpp_Z"] = metrics_z["Bpp"]
        metrics["PSNR_cmpref_Z"] = metrics_z["PSNR_cmpref"]

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        out_dir / "curves.npz",
        loss_values=np.asarray(loss_h),
        bpp_values=np.asarray(bpp_h),
        psnr_values=np.asarray(psnr_h),
    )
    payload = {"sr_final": sr_final.detach().cpu()}
    if Z is not None:
        payload["Z_tensor"] = Z.detach().cpu()
        payload["T_tensor"] = T.detach().cpu()
        tensor_to_pil(Z).save(out_dir / "Z.png")
    torch.save(payload, out_dir / "Z.pt")
    tensor_to_pil(sr_final).save(out_dir / "SR.png")
    tensor_to_pil(gt).save(out_dir / "GT.png")

    keys = ("method", "PSNR", "Bpp", "Bpp(fsize)", "PSNR_cmpref", "SSI_cmpref", "bpp_train_end", "bpp_train_min")
    print("metrics:", json.dumps({k: metrics[k] for k in keys}, indent=2))
    print(f"Saved -> {out_dir}")
    return metrics


def build_argparser():
    p = argparse.ArgumentParser(description="Bitrate-SR LoRA (admm|direct; nina|swin)")
    p.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    p.add_argument("--data-dir", default=None)
    p.add_argument("--runs-dir", default=None, help="Output root (default: PROJECT/runs/bitrate_sr)")
    p.add_argument("--img-list", required=True, help="Txt with one DIV2K filename per line")
    p.add_argument("--lambda", dest="lam", type=float, required=True)
    p.add_argument("--method", default="admm", choices=["admm", "direct"])
    p.add_argument(
        "--direct-ref",
        default="gt",
        choices=["gt", "sr_orig"],
        help="Direct method: PSNR hinges vs GT or pretrained SR",
    )
    p.add_argument("--steps", type=int, default=1000, help="Direct method: Adam steps")
    p.add_argument("--checkpoint-every", type=int, default=200, help="Direct: save every N steps (0=off)")
    p.add_argument("--target-psnr", type=float, default=35.0)
    p.add_argument("--backbone", default="nina", choices=["nina", "swin"])
    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=float, default=8.0)
    p.add_argument(
        "--lora-target",
        default="all",
        choices=["all", "attention_expand"],
        help="Nina: all Conv2d LoRA, or only body.*.body.2.body.3",
    )
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--inners", type=int, default=100)
    p.add_argument("--outers", type=int, default=20)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--scaled-size", type=int, default=256)
    p.add_argument("--crop-left", type=int, default=1328)
    p.add_argument("--crop-top", type=int, default=826)
    p.add_argument("--device", default="")
    p.add_argument("--compression-model", default="cheng2020-attn")
    p.add_argument("--compression-quality", type=int, default=6)
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    img_list_path = Path(args.img_list)
    if not img_list_path.is_file():
        print(f"--img-list not found: {img_list_path}", file=sys.stderr)
        return 2

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading compression on {device} ... method={args.method}")
    compression = build_compression(device, args.compression_model, args.compression_quality)

    images = list(dict.fromkeys(parse_img_list(img_list_path)))
    if not images:
        print(f"Empty image list: {img_list_path}", file=sys.stderr)
        return 2

    all_metrics = [run_one(args, name, compression) for name in images]

    if len(all_metrics) > 1:
        runs_dir = Path(args.runs_dir) if args.runs_dir else Path(args.project_root) / "runs" / "bitrate_sr"
        method = (args.method or "admm").lower()
        lt = (args.lora_target or "all").lower()
        lt_tag = "" if lt in ("all", "") else f"_{ {'attention_expand': 'attnexp'}.get(lt, lt)}"
        summary = runs_dir / f"summary_{method}_{args.backbone}_lam{args.lam:g}_psnr{args.target_psnr:g}{lt_tag}.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(all_metrics, indent=2))
        print(f"Summary -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
