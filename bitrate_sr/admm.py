"""ADMM loop: minimize bitrate under PSNR constraints via LoRA-SR."""

from __future__ import annotations

from pathlib import Path

import torch

from .metrics import bpp_loss, forward_sr, psnr


def _save_ckpt(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    torch.save(state, path.parent / "latest.pt")


def _project_psnr_ball(d: torch.Tensor, center: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of d onto {z : ||z - center||_2 <= delta}."""
    e = d - center
    n_e = torch.linalg.norm(e.reshape(-1))
    if n_e <= delta:
        return d.detach()
    return (center + delta * e / n_e).detach()


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
    z_init: torch.Tensor | None = None,
    t_init: torch.Tensor | None = None,
    project_z_init: bool = True,
):
    """ADMM run. Optional warm-start: z_init / t_init (e.g. from a direct SR run).

    Returns model, Z, T, loss/bpp/psnr histories.
    """
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

    if z_init is None:
        Z = sr_orig.clone()
    else:
        Z = z_init.to(device=device, dtype=sr_orig.dtype)
        if project_z_init:
            Z = _project_psnr_ball(Z, sr_orig, delta)
    T = (
        torch.zeros_like(sr_orig)
        if t_init is None
        else t_init.to(device=device, dtype=sr_orig.dtype).detach()
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

    # CompressAI: train() uses noise for differentiable rate; eval() quantizes (no bpp grads).
    # Weights stay frozen (requires_grad=False); we only need the train forward path.
    was_training = model_compression.training
    model_compression.train()
    model_supres.train()

    ki = 0
    sr = None
    try:
        for ki in range(outer_iterations):
            ZT = Z + T
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
            Z = _project_psnr_ball(D, sr_orig, delta)
            T = (T + Z - sr).detach()

            if checkpoint_dir is not None:
                _save_ckpt(pack_state(ki + 1), Path(checkpoint_dir) / f"outer_{ki:04d}.pt")
                print(f"Checkpoint: outer {ki + 1}/{outer_iterations}")

    except KeyboardInterrupt:
        if checkpoint_dir is not None:
            _save_ckpt(pack_state(ki + 1), Path(checkpoint_dir) / "interrupt.pt")
            print(f"Interrupted -> {Path(checkpoint_dir) / 'latest.pt'}")
    finally:
        model_compression.train(was_training)
        model_supres.eval()

    return model_supres, Z, T, loss_h, bpp_h, psnr_h
