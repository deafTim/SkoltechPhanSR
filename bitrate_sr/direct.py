"""Direct one-loss bitrate minimization (no ADMM Z/T)."""

from __future__ import annotations

from pathlib import Path

import torch

from .metrics import bpp_loss, forward_sr, psnr


def _save_ckpt(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    torch.save(state, path.parent / "latest.pt")


def minbitrate_superresolution_direct(
    input_tensor,
    gt_tensor,
    sr_orig,
    target_psnr: float,
    quality_loss_lambda: float,
    model_supres,
    model_compression,
    *,
    lr: float = 5e-4,
    steps: int = 1000,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 200,
    ref: str = "gt",
):
    """Adam on LoRA with a single rate + PSNR-hinge loss.

    loss = 10*log10(bpp) + λ * (q_cmp + q_sr)
    q_* = max(|PSNR - target| - 0.5, 0)

    ref: "gt" → constraints vs ground truth (matches RD metrics);
         "sr_orig" → vs pretrained SR (closer to ADMM ablation).
    """
    device = input_tensor.device
    if sr_orig is None:
        with torch.no_grad():
            sr_orig = forward_sr(model_supres, input_tensor).detach()

    if ref == "gt":
        quality_ref = gt_tensor
    elif ref == "sr_orig":
        quality_ref = sr_orig
    else:
        raise ValueError(f"Unknown ref={ref!r} (use gt|sr_orig)")

    max_i = 1.0
    n_pix = sr_orig.shape[0] * sr_orig.shape[2] * sr_orig.shape[3]

    params = [p for p in model_supres.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters (LoRA inject failed?)")
    opt = torch.optim.Adam(params, lr=lr)

    loss_h, bpp_h, psnr_h = [], [], []

    was_training = model_compression.training
    model_compression.train()
    model_supres.train()

    sr = None
    try:
        for t in range(steps):
            opt.zero_grad()
            sr = forward_sr(model_supres, input_tensor)
            comp = model_compression(sr)
            bpploss = bpp_loss(comp, n_pix)

            psnr_cmp = psnr(comp["x_hat"], quality_ref, max_i)
            psnr_sr = psnr(sr, quality_ref, max_i)
            zero = torch.zeros((), device=device, dtype=sr.dtype)
            q1 = torch.maximum((psnr_cmp - target_psnr).abs() - 0.5, zero)
            q2 = torch.maximum((psnr_sr - target_psnr).abs() - 0.5, zero)
            loss = 10 * torch.log10(bpploss + 1e-8) + quality_loss_lambda * (q1 + q2)
            loss.backward()
            opt.step()

            if t % 30 == 0:
                print(
                    f"Step {t}/{steps} | PSNR cmp: {psnr_cmp:.2f} | PSNR sr: {psnr_sr:.2f} "
                    f"| BPP: {bpploss:.4f} | Loss: {loss:.4f}"
                )
            loss_h.append(float(loss.item()))
            bpp_h.append(float(bpploss.item()))
            psnr_h.append(float(psnr_cmp.item()))

            if checkpoint_dir is not None and checkpoint_every > 0 and (t + 1) % checkpoint_every == 0:
                _save_ckpt(
                    {
                        "model_supres": model_supres.state_dict(),
                        "step": t + 1,
                        "optimizer": opt.state_dict(),
                        "loss_values": loss_h,
                        "bpp_values": bpp_h,
                        "psnr_values": psnr_h,
                        "target_psnr": float(target_psnr),
                        "quality_loss_lambda": float(quality_loss_lambda),
                        "steps": int(steps),
                        "learningrate": float(lr),
                        "method": "direct",
                        "ref": ref,
                    },
                    Path(checkpoint_dir) / f"step_{t + 1:04d}.pt",
                )
                print(f"Checkpoint: step {t + 1}/{steps}")

    except KeyboardInterrupt:
        if checkpoint_dir is not None:
            _save_ckpt(
                {
                    "model_supres": model_supres.state_dict(),
                    "step": len(loss_h),
                    "optimizer": opt.state_dict(),
                    "loss_values": loss_h,
                    "bpp_values": bpp_h,
                    "psnr_values": psnr_h,
                    "method": "direct",
                    "ref": ref,
                },
                Path(checkpoint_dir) / "interrupt.pt",
            )
            print(f"Interrupted -> {Path(checkpoint_dir) / 'latest.pt'}")
    finally:
        model_compression.train(was_training)
        model_supres.eval()

    return model_supres, sr.detach() if sr is not None else None, loss_h, bpp_h, psnr_h
