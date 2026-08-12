#!/usr/bin/env python3
"""Collect metrics.json from runs/bitrate_sr and plot RD / print table.

Prefer: edit scripts/bitrate_sr_config.sh, then bash scripts/plot_results.sh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import DEFAULT_PROJECT_ROOT


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs-dir",
        type=str,
        default=str(DEFAULT_PROJECT_ROOT / "runs" / "bitrate_sr"),
    )
    ap.add_argument("--save", type=str, default="")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.runs_dir)
    rows = []
    for d in sorted(root.glob("img*_psnr*_lam*")):
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text())
        m["_dir"] = d.name
        rows.append(m)

    if not rows:
        print(f"No metrics found under {root}")
        return 1

    print(f"{'dir':45s} {'lam':>6s} {'Bpp':>8s} {'PSNRref':>8s} {'SSIMref':>8s}")
    for m in rows:
        print(
            f"{m.get('_dir', ''):45s} {m.get('lambda', float('nan')):6.3g} "
            f"{m.get('Bpp', float('nan')):8.4f} {m.get('PSNR_cmpref', float('nan')):8.3f} "
            f"{str(m.get('SSI_cmpref')):>8s}"
        )

    xs = [m.get("Bpp") for m in rows if m.get("Bpp") is not None]
    ys = [m.get("PSNR_cmpref") for m in rows if m.get("PSNR_cmpref") is not None]
    labels = [f"λ={m.get('lambda')}" for m in rows]

    if xs and ys and len(xs) == len(ys):
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        labels = [labels[i] for i in order]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, "o-")
        for x, y, lab in zip(xs, ys, labels):
            plt.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        plt.xlabel("Bpp")
        plt.ylabel("PSNR vs GT (compressed)")
        plt.title("Bitrate-SR ADMM-LoRA rate–distortion")
        plt.grid(True, alpha=0.3)
        if args.save:
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"Saved plot -> {args.save}")
        if args.show:
            plt.show()
        else:
            plt.close()

    cpath = root / rows[0]["_dir"] / "curves.npz"
    if cpath.exists():
        c = np.load(cpath)
        print(f"Example curves in {cpath}: keys={list(c.keys())}, bpp len={len(c['bpp_values'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
