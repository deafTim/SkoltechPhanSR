#!/usr/bin/env python3
"""Bitrate-minimizing SR with LoRA (NinaSR or Swin2SR).

Methods: ADMM (default), direct one-loss Adam, or admm_oldlike
(old-notebook formulation, RD swept via a target-PSNR ladder).
ADMM can warm-start from a finished direct run (--admm-init direct).
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
from .admm_oldlike import minbitrate_superresolution_oldlike
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


def _method_tag(args) -> str:
    method = (args.method or "admm").lower()
    if method == "admm_oldlike":
        return "admm_old"
    if method == "admm" and (getattr(args, "admm_init", "pretrained") or "pretrained") == "direct":
        return "admm_wd"
    return method


def _parse_psnr_ladder(args) -> list[float]:
    """Target-PSNR rungs for admm_oldlike (e.g. "36 35 34 33 32")."""
    raw = (getattr(args, "anneal_psnrs", None) or "").replace(",", " ").split()
    if not raw:
        return [float(args.target_psnr)]
    return [float(v) for v in raw]


def _load_direct_warmstart(
    args,
    img_name: str,
    model,
    device: torch.device,
) -> tuple[torch.Tensor | None, bool]:
    """Load SR (and optional LoRA weights) from a matching direct run.

    Returns (z_init, loaded_weights).
    """
    init_root = getattr(args, "admm_init_runs_dir", None) or ""
    if not init_root:
        raise ValueError("--admm-init direct requires --admm-init-runs-dir")
    init_root = Path(init_root)
    lora_target = (getattr(args, "lora_target", None) or "all").lower()
    dtag = run_tag(
        Path(img_name).stem,
        args.target_psnr,
        args.lam,
        args.backbone,
        args.lora_r,
        method="direct",
        lora_target=lora_target,
    )
    ddir = init_root / dtag
    if not ddir.is_dir():
        raise FileNotFoundError(f"direct init dir not found: {ddir}")

    loaded_weights = False
    for ckpt_name in ("latest.pt", "interrupt.pt"):
        ckpt_path = ddir / ckpt_name
        if not ckpt_path.is_file():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_supres")
        if state is not None:
            model.load_state_dict(state, strict=False)
            loaded_weights = True
            print(f"Warm-start LoRA from {ckpt_path}")
            break

    z_init = None
    zpt = ddir / "Z.pt"
    if zpt.is_file():
        payload = torch.load(zpt, map_location="cpu", weights_only=False)
        sr = payload.get("sr_final")
        if sr is not None:
            z_init = sr.to(device)
            print(f"Warm-start Z from {zpt} (sr_final)")
    if z_init is None:
        sr_png = ddir / "SR.png"
        if sr_png.is_file():
            from PIL import Image
            from torchvision import transforms

            t = transforms.ToTensor()(Image.open(sr_png).convert("RGB")).unsqueeze(0).to(device)
            z_init = t
            print(f"Warm-start Z from {sr_png}")
    if z_init is None:
        raise FileNotFoundError(f"no sr_final/SR.png in {ddir}")
    if not loaded_weights:
        print(f"Warning: no LoRA ckpt in {ddir}; only Z warm-started (fresh LoRA)")
    return z_init, loaded_weights


def _save_run(
    out_dir: Path,
    tag: str,
    tag_method: str,
    model,
    model_compression,
    sr_final: torch.Tensor,
    gt: torch.Tensor,
    Z: torch.Tensor | None,
    T: torch.Tensor | None,
    extra: dict,
    loss_h: list,
    bpp_h: list,
    psnr_h: list,
) -> dict:
    """Evaluate + write metrics.json, curves, Z/SR/GT images and latest.pt."""
    metrics = eval_perf_compression(model_compression, sr_final, gt)
    metrics.update(extra)
    metrics["tag"] = tag
    metrics["bpp_train_end"] = float(bpp_h[-1]) if bpp_h else None
    metrics["bpp_train_min"] = float(min(bpp_h)) if bpp_h else None
    if Z is not None:
        metrics_z = eval_perf_compression(model_compression, Z, gt)
        metrics["Bpp_Z"] = metrics_z["Bpp"]
        metrics["PSNR_cmpref_Z"] = metrics_z["PSNR_cmpref"]

    out_dir.mkdir(parents=True, exist_ok=True)
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
    torch.save(
        {"model_supres": model.state_dict(), "method": tag_method, "tag": tag},
        out_dir / "latest.pt",
    )
    tensor_to_pil(sr_final).save(out_dir / "SR.png")
    tensor_to_pil(gt).save(out_dir / "GT.png")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _load_rung_state(out_dir: Path, model, device) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Reload LoRA + Z/T of a finished rung so the ladder can continue after a restart."""
    zpt = out_dir / "Z.pt"
    if not zpt.is_file():
        return None
    payload = torch.load(zpt, map_location="cpu", weights_only=False)
    Z = payload.get("Z_tensor")
    T = payload.get("T_tensor")
    if Z is None or T is None:
        return None
    ckpt_path = out_dir / "latest.pt"
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_supres")
        if state is not None:
            model.load_state_dict(state, strict=False)
    return Z.to(device), T.to(device)


def run_one_oldlike(args, img_name: str, model_compression) -> dict:
    """Old-notebook ADMM: one job walks the target-PSNR ladder, warm-starting each rung."""
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    project = Path(args.project_root)
    data_dir = Path(args.data_dir) if args.data_dir else project / "DIV2K" / "DIV2K_train_HR"
    runs_dir = Path(args.runs_dir) if args.runs_dir else project / "runs" / "bitrate_sr"

    img_path = data_dir / img_name
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    lora_target = (getattr(args, "lora_target", None) or "all").lower()
    tag_method = "admm_old"
    ladder = _parse_psnr_ladder(args)
    tags = [
        run_tag(
            Path(img_name).stem,
            tp,
            args.lam,
            args.backbone,
            args.lora_r,
            method=tag_method,
            lora_target=lora_target,
        )
        for tp in ladder
    ]
    if all((runs_dir / t / "metrics.json").is_file() for t in tags):
        print(f"=== skip {img_name} ({tag_method}: all {len(tags)} rungs done) ===")
        return json.loads((runs_dir / tags[-1] / "metrics.json").read_text())

    print(
        f"=== {img_name} {tag_method} === ladder={ladder} lam={args.lam:g} "
        f"repeats={args.repeats} outers={args.outers} inners={args.inners} lr={args.lr:g} "
        f"device={device} runs_dir={runs_dir}"
    )

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
    # Ball center stays the pretrained SR for every rung, as in the notebook.
    with torch.no_grad():
        sr_orig = forward_sr(model, x).detach()

    Z = T = None
    last_metrics: dict = {}
    for tp, tag in zip(ladder, tags):
        out_dir = runs_dir / tag
        if (out_dir / "metrics.json").is_file():
            resumed = _load_rung_state(out_dir, model, device)
            if resumed is not None:
                Z, T = resumed
                last_metrics = json.loads((out_dir / "metrics.json").read_text())
                print(f"--- skip psnr={tp:g} (exists), ladder resumed from {tag}")
                continue
            print(f"--- {tag} exists but Z/T unreadable; recomputing this rung")

        print(f"--- rung psnr={tp:g} -> {out_dir}")
        model, Z, T, loss_h, bpp_h, psnr_h = minbitrate_superresolution_oldlike(
            x,
            sr_orig,
            tp,
            args.lam,
            args.inners,
            model,
            model_compression,
            lr=args.lr,
            outer_iterations=args.outers,
            repeats=args.repeats,
            checkpoint_dir=out_dir if args.checkpoint_every > 0 else None,
            z_init=Z,
            t_init=T,
        )
        model.eval()
        with torch.no_grad():
            sr_final = forward_sr(model, x).detach()

        last_metrics = _save_run(
            out_dir,
            tag,
            tag_method,
            model,
            model_compression,
            sr_final,
            gt,
            Z,
            T,
            {
                "img": img_name,
                "backbone": args.backbone,
                "method": tag_method,
                "admm_init": "ladder",
                "lora_target": lora_target,
                "lambda": args.lam,
                "target_psnr": tp,
                "anneal_psnrs": ladder,
                "repeats": args.repeats,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lr": args.lr,
                "inners": args.inners,
                "outers": args.outers,
                "steps": args.repeats * args.outers * args.inners,
                "direct_ref": None,
                "crop": crop_meta,
            },
            loss_h,
            bpp_h,
            psnr_h,
        )
        keys = ("method", "target_psnr", "PSNR", "Bpp", "PSNR_cmpref", "Bpp_Z", "PSNR_cmpref_Z")
        print("rung metrics:", json.dumps({k: last_metrics.get(k) for k in keys}, indent=2))
        print(f"Saved -> {out_dir}")

    return last_metrics


def run_one(args, img_name: str, model_compression) -> dict:
    if (args.method or "").lower() == "admm_oldlike":
        return run_one_oldlike(args, img_name, model_compression)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    project = Path(args.project_root)
    data_dir = Path(args.data_dir) if args.data_dir else project / "DIV2K" / "DIV2K_train_HR"
    runs_dir = Path(args.runs_dir) if args.runs_dir else project / "runs" / "bitrate_sr"

    img_path = data_dir / img_name
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    method = (args.method or "admm").lower()
    lora_target = (getattr(args, "lora_target", None) or "all").lower()
    admm_init = (getattr(args, "admm_init", None) or "pretrained").lower()
    tag_method = _method_tag(args)
    tag = run_tag(
        Path(img_name).stem,
        args.target_psnr,
        args.lam,
        args.backbone,
        args.lora_r,
        method=tag_method,
        lora_target=lora_target,
    )
    out_dir = runs_dir / tag
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file():
        print(f"=== skip {tag} (exists {metrics_path}) ===")
        return json.loads(metrics_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"=== {tag} ===\nmethod={method} admm_init={admm_init} "
        f"lora_target={lora_target} device={device} out_dir={out_dir}"
    )

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

    z_init = t_init = None
    if method == "admm" and admm_init == "direct":
        z_init, _ = _load_direct_warmstart(args, img_name, model, device)
        # After loading LoRA from direct, refresh sr_orig for ball center:
        # keep pretrained ball (sr_orig before load). We already computed sr_orig
        # on fresh model — that is correct (ball around pretrained SR).
        # Recompute forward after weight load only for logging.
        with torch.no_grad():
            sr_after = forward_sr(model, x).detach()
        from .metrics import psnr as _psnr

        print(f"Warm-start ready: PSNR(loaded_SR, sr_orig)={float(_psnr(sr_after, sr_orig).cpu()):.2f}")

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
            checkpoint_dir=out_dir if args.checkpoint_every > 0 else None,
            z_init=z_init,
            t_init=t_init,
            project_z_init=True,
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
            "method": tag_method,
            "admm_init": admm_init if method == "admm" else None,
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
    # Always keep LoRA weights for warm-start / resume (even if step checkpoints off)
    torch.save(
        {
            "model_supres": model.state_dict(),
            "method": tag_method,
            "tag": tag,
        },
        out_dir / "latest.pt",
    )
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
    p.add_argument(
        "--img-index",
        type=int,
        default=-1,
        help="Run only this line of --img-list (-1 = all); used by SLURM arrays",
    )
    p.add_argument("--lambda", dest="lam", type=float, required=True)
    p.add_argument("--method", default="admm", choices=["admm", "direct", "admm_oldlike"])
    p.add_argument(
        "--admm-init",
        default="pretrained",
        choices=["pretrained", "direct"],
        help="ADMM only: init Z (and LoRA if latest.pt exists) from a direct run",
    )
    p.add_argument(
        "--admm-init-runs-dir",
        default="",
        help="Runs root with finished *_direct tags (required if --admm-init direct)",
    )
    p.add_argument(
        "--direct-ref",
        default="gt",
        choices=["gt", "sr_orig"],
        help="Direct method: PSNR hinges vs GT or pretrained SR",
    )
    p.add_argument("--steps", type=int, default=1000, help="Direct method: Adam steps")
    p.add_argument("--checkpoint-every", type=int, default=200, help="Direct: save every N steps (0=off)")
    p.add_argument("--target-psnr", type=float, default=35.0)
    p.add_argument(
        "--anneal-psnrs",
        default="",
        help="admm_oldlike: target-PSNR ladder, e.g. '36 35 34 33 32' (empty = --target-psnr)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="admm_oldlike: Adam restarts per rung (krun in the notebook)",
    )
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
    if (args.method or "").lower() == "admm" and (args.admm_init or "") == "direct":
        if not args.admm_init_runs_dir:
            print("--admm-init direct requires --admm-init-runs-dir", file=sys.stderr)
            return 2

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(
        f"Loading compression on {device} ... method={args.method} "
        f"admm_init={getattr(args, 'admm_init', 'pretrained')}"
    )
    compression = build_compression(device, args.compression_model, args.compression_quality)

    images = list(dict.fromkeys(parse_img_list(img_list_path)))
    if not images:
        print(f"Empty image list: {img_list_path}", file=sys.stderr)
        return 2
    if args.img_index >= 0:
        if args.img_index >= len(images):
            print(
                f"--img-index {args.img_index} out of range (list has {len(images)})",
                file=sys.stderr,
            )
            return 2
        images = [images[args.img_index]]
        print(f"img-index {args.img_index} -> {images[0]}")

    all_metrics = [run_one(args, name, compression) for name in images]

    if len(all_metrics) > 1:
        runs_dir = Path(args.runs_dir) if args.runs_dir else Path(args.project_root) / "runs" / "bitrate_sr"
        tag_method = _method_tag(args)
        lt = (args.lora_target or "all").lower()
        lt_tag = "" if lt in ("all", "") else f"_{ {'attention_expand': 'attnexp'}.get(lt, lt)}"
        summary = runs_dir / f"summary_{tag_method}_{args.backbone}_lam{args.lam:g}_psnr{args.target_psnr:g}{lt_tag}.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(all_metrics, indent=2))
        print(f"Summary -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
