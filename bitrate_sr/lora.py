"""LoRA wrappers: Conv2d (NinaSR) and Linear (Swin2SR)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    """Frozen base Conv2d + trainable low-rank path."""

    def __init__(self, conv: nn.Conv2d, r: int = 4, alpha: float = 8.0):
        super().__init__()
        if conv.groups != 1:
            raise ValueError("LoRAConv2d supports groups=1 only")
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False
        self.scaling = alpha / r
        self.lora_A = nn.Conv2d(
            conv.in_channels,
            r,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=False,
        )
        self.lora_B = nn.Conv2d(r, conv.out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.conv(x) + self.lora_B(self.lora_A(x)) * self.scaling


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank path (for Swin attention)."""

    def __init__(self, linear: nn.Linear, r: int = 4, alpha: float = 8.0):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base = self.linear(x)
        # x @ A^T @ B^T * scale
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora


def inject_lora_conv2d(module: nn.Module, r: int = 4, alpha: float = 8.0) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (LoRAConv2d, LoRALinear)):
            continue
        if isinstance(child, nn.Conv2d):
            setattr(module, name, LoRAConv2d(child, r=r, alpha=alpha))
            n += 1
        else:
            n += inject_lora_conv2d(child, r=r, alpha=alpha)
    return n


# Typical Swin2SR / HF attention Linear names
_SWIN_LINEAR_HINTS = (
    "query",
    "key",
    "value",
    "qkv",
    "proj",
    "dense",
    "fc1",
    "fc2",
)


def _linear_name_ok(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in _SWIN_LINEAR_HINTS)


def inject_lora_linear(
    module: nn.Module,
    r: int = 4,
    alpha: float = 8.0,
    name_filter=_linear_name_ok,
) -> int:
    """Wrap attention/MLP Linears with LoRALinear. Returns #replaced."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, (LoRALinear, LoRAConv2d)):
            continue
        if isinstance(child, nn.Linear) and name_filter(name):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            n += 1
        else:
            n += inject_lora_linear(child, r=r, alpha=alpha, name_filter=name_filter)
    return n
