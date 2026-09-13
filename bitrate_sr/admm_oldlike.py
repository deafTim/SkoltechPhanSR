"""ADMM as formulated in train_superresolution_compression_admm_div2K_0025_old.ipynb.

Kept separate from `admm.py` on purpose; the differences are the ones that made
the notebook hold the PSNR constraint:

  * plain |PSNR - target| penalties (no 0.5 dB dead zone)
  * the SR-vs-(Z+T) term is weighted lambda/10, not lambda
  * a fresh Adam is created for every `repeats` restart, lr 1e-4 by default

The rate/quality trade-off is swept via lambda at a fixed target_psnr (same
as the regular ADMM sweeps). Optional `--anneal-psnrs` still supports a
target-PSNR ladder if needed.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .admm import _project_psnr_ball, _save_ckpt
from .metrics import bpp_loss, forward_sr, psnr


def minbitrate_superresolution_oldlike(
    input_tensor,
    sr_orig,
    target_psnr: float,
    quality_loss_lambda: float,
    num_iterations: int,
    model_supres,
    model_compression,
    *,
    lr: float = 1e-4,
    outer_iterations: int = 50,
    repeats: int = 4,
    conv_tol: float = 1e-4,
    checkpoint_dir: Path | None = None,
    z_init: torch.Tensor | None = None,
    t_init: torch.Tensor | None = None,
    log_every: int = 10,
):
    """One target-PSNR rung. Returns model, Z, T, loss/bpp/psnr histories."""
    device = input_tensor.device
    if sr_orig is None:
        with torch.no_grad():
            sr_orig = forward_sr(model_supres, input_tensor).detach()

    max_i = 1.0
    n_pix = sr_orig.shape[0] * sr_orig.shape[2] * sr_orig.shape[3]
    delta = torch.sqrt(
        torch.tensor(
            sr_orig.shape[1] * n_pix * (max_i ** 2) / (10 ** (target_psnr / 10)),
            device=device,
            dtype=sr_orig.dtype,
        )
    )

    Z = (
        sr_orig.clone()
        if z_init is None
        else z_init.to(device=device, dtype=sr_orig.dtype).detach()
    )
    T = (
        torch.zeros_like(sr_orig)
        if t_init is None
        else t_init.to(device=device, dtype=sr_orig.dtype).detach()
    )

    params = [p for p in model_supres.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters (LoRA inject failed?)")

    loss_h, bpp_h, psnr_h = [], [], []

    def pack_state(krun: int, outer_idx: int, opt) -> dict:
        return {
            "model_supres": model_supres.state_dict(),
            "Z_tensor": Z.detach().cpu(),
            "T_tensor": T.detach().cpu(),
            "repeat_index": krun,
            "outer_index": outer_idx,
            "optimizer": opt.state_dict(),
            "loss_values": loss_h,
            "bpp_values": bpp_h,
            "psnr_values": psnr_h,
            "target_psnr": float(target_psnr),
            "quality_loss_lambda": float(quality_loss_lambda),
            "num_iterations": int(num_iterations),
            "learningrate": float(lr),
            "method": "admm_old",
        }

    # CompressAI: train() uses noise for a differentiable rate estimate.
    was_training = model_compression.training
    model_compression.train()
    model_supres.train()

    opt = torch.optim.Adam(params, lr=lr)
    krun = ki = 0
    sr = None
    try:
        for krun in range(repeats):
            # The notebook called the function anew per krun, so Adam restarts too.
            opt = torch.optim.Adam(params, lr=lr)
            for ki in range(outer_iterations):
                ZT = Z + T
                for it in range(num_iterations):
                    opt.zero_grad()
                    sr = forward_sr(model_supres, input_tensor)
                    comp = model_compression(sr)
                    bpploss = bpp_loss(comp, n_pix)

                    psnr_cmp = psnr(comp["x_hat"], sr_orig, max_i)
                    psnr_sr = psnr(sr, ZT, max_i)
                    loss = (
                        10 * torch.log10(bpploss + 1e-8)
                        + quality_loss_lambda * (psnr_cmp - target_psnr).abs()
                        + quality_loss_lambda / 10 * (psnr_sr - target_psnr).abs()
                    )
                    loss.backward()
                    opt.step()

                    if log_every and it % log_every == 0:
                        print(
                            f"t{target_psnr:g} | run {krun} | outer {ki} | iter {it} "
                            f"| PSNR cmp: {psnr_cmp:.2f} | PSNR sr: {psnr_sr:.2f} "
                            f"| BPP: {bpploss:.4f} | Loss: {loss:.4f}"
                        )
                    loss_h.append(float(loss.item()))
                    bpp_h.append(float(bpploss.item()))
                    psnr_h.append(float(psnr_sr.item()))

                D = sr - T
                Z = _project_psnr_ball(D, sr_orig, delta)
                T = (T + Z - sr).detach()

                if checkpoint_dir is not None:
                    _save_ckpt(
                        pack_state(krun, ki + 1, opt),
                        Path(checkpoint_dir) / f"run{krun:02d}_outer_{ki:04d}.pt",
                    )
                    print(f"Checkpoint: run {krun} outer {ki + 1}/{outer_iterations}")

            if len(loss_h) > 1 and abs(loss_h[-1] - loss_h[-2]) < conv_tol:
                print(f"Converged at target {target_psnr:g} dB (run {krun + 1}/{repeats})")
                break

    except KeyboardInterrupt:
        if checkpoint_dir is not None:
            _save_ckpt(pack_state(krun, ki + 1, opt), Path(checkpoint_dir) / "interrupt.pt")
            print(f"Interrupted -> {Path(checkpoint_dir) / 'latest.pt'}")
    finally:
        model_compression.train(was_training)
        model_supres.eval()

    return model_supres, Z, T, loss_h, bpp_h, psnr_h
