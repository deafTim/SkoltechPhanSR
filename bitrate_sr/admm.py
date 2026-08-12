"""ADMM loop: minimize bitrate under PSNR constraints via LoRA-SR."""

from __future__ import annotations

from pathlib import Path

import torch

from .metrics import bpp_loss, forward_sr, psnr


def _save_ckpt(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    torch.save(state, path.parent / "latest.pt")


def minbitrate_superresolution(
    input_tensor,
    sr_orig,
    target_psnr: float,
    quality_loss_lambda: float,
    num_iterations: int,
    model_supres,
    model_compression,
    *,
    lr: float = 5e-4,
    outer_iterations: int = 20,
    checkpoint_dir: Path | None = None,
):
    """Fresh ADMM run. Returns model, Z, T, loss/bpp/psnr histories."""
    device = input_tensor.device
    if sr_orig is None:
        with torch.no_grad():
            sr_orig = forward_sr(model_supres, input_tensor).detach()

    max_i = 1.0
    n_pix = sr_orig.shape[0] * sr_orig.shape[2] * sr_orig.shape[3]
    Z = sr_orig.clone()
    T = torch.zeros_like(sr_orig)
    delta = torch.sqrt(
        torch.tensor(
            sr_orig.shape[1] * n_pix * (max_i ** 2) / (10 ** (target_psnr / 10)),
            device=device,
            dtype=sr_orig.dtype,
        )
    )

    params = [p for p in model_supres.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters (LoRA inject failed?)")
    opt = torch.optim.Adam(params, lr=lr)

    loss_h, bpp_h, psnr_h = [], [], []

    def pack_state(outer_idx: int) -> dict:
        return {
            "model_supres": model_supres.state_dict(),
            "Z_tensor": Z.detach().cpu(),
            "T_tensor": T.detach().cpu(),
            "outer_index": outer_idx,
            "optimizer": opt.state_dict(),
            "loss_values": loss_h,
            "bpp_values": bpp_h,
            "psnr_values": psnr_h,
            "target_psnr": float(target_psnr),
            "quality_loss_lambda": float(quality_loss_lambda),
            "num_iterations": int(num_iterations),
            "learningrate": float(lr),
        }

    ki = 0
    try:
        for ki in range(outer_iterations):
            ZT = Z + T
            sr = None
            for it in range(num_iterations):
                opt.zero_grad()
                sr = forward_sr(model_supres, input_tensor)
                comp = model_compression(sr)
                bpploss = bpp_loss(comp, n_pix)

                psnr_cmp = psnr(comp["x_hat"], sr_orig, max_i)
                psnr_sr = psnr(sr, ZT, max_i)
                zero = torch.zeros((), device=device, dtype=sr.dtype)
                q1 = torch.maximum((psnr_cmp - target_psnr).abs() - 0.5, zero)
                q2 = torch.maximum((psnr_sr - target_psnr).abs() - 0.5, zero)
                loss = 10 * torch.log10(bpploss + 1e-8) + quality_loss_lambda * (q1 + q2)
                loss.backward()
                opt.step()

                if it % 30 == 0:
                    print(
                        f"Outer {ki} | Iter {it} | PSNR cmp: {psnr_cmp:.2f} "
                        f"| BPP: {bpploss:.4f} | Loss: {loss:.4f}"
                    )
                loss_h.append(float(loss.item()))
                bpp_h.append(float(bpploss.item()))
                psnr_h.append(float(psnr_sr.item()))

            D = sr - T
            E = D - sr_orig
            nE = torch.linalg.norm(E.reshape(-1))
            Z = (D if nE <= delta else sr_orig + delta * E / nE).detach()
            T = (T + Z - sr).detach()

            if checkpoint_dir is not None:
                _save_ckpt(pack_state(ki + 1), Path(checkpoint_dir) / f"outer_{ki:04d}.pt")
                print(f"Checkpoint: outer {ki + 1}/{outer_iterations}")

    except KeyboardInterrupt:
        if checkpoint_dir is not None:
            _save_ckpt(pack_state(ki + 1), Path(checkpoint_dir) / "interrupt.pt")
            print(f"Interrupted -> {Path(checkpoint_dir) / 'latest.pt'}")

    return model_supres, Z, T, loss_h, bpp_h, psnr_h
