"""Lightweight EO-SAR change detection models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module: nn.Module) -> None:
    """Kaiming initialization for convolutional segmentation networks."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ConvBlock(nn.Module):
    """Two small convolutions with normalization and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsample then apply a convolution block."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample, concatenate skip features, and refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class LightweightUNet(nn.Module):
    """Compact early-fusion U-Net for 4-channel EO+SAR inputs."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 1,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.05,
    ):
        super().__init__()
        channels = [base_channels * mult for mult in channel_multipliers]
        self.stem = ConvBlock(in_channels, channels[0], dropout=0.0)
        self.downs = nn.ModuleList(
            DownBlock(channels[i], channels[i + 1], dropout)
            for i in range(len(channels) - 1)
        )
        self.ups = nn.ModuleList(
            UpBlock(channels[i], channels[i - 1], channels[i - 1], dropout)
            for i in range(len(channels) - 1, 0, -1)
        )
        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return self.head(x)


class EOSARChangeDetector(nn.Module):
    """Backward-compatible model wrapper used by scripts and checkpoints."""

    def __init__(
        self,
        encoder_name: str = "lightunet",
        encoder_weights: str | None = None,
        in_channels: int = 4,
        num_classes: int = 1,
        base_channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.05,
    ):
        super().__init__()
        if encoder_weights is not None:
            raise ValueError("The lightweight local model does not use pretrained weights.")
        if encoder_name not in {"lightunet", "fc_ef", "unet"}:
            raise ValueError(
                f"Unsupported lightweight model '{encoder_name}'. Use lightunet/fc_ef/unet."
            )
        self.model = LightweightUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_model_state_flexible(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load repo or notebook checkpoints into the wrapped model.

    The Kaggle notebook saved ``EOSARChangeDetector`` directly, with keys such
    as ``stem.block.0.weight``. The structured repo wraps that same network in
    ``self.model``, producing keys prefixed with ``model.``.
    """
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    model_keys = model.state_dict().keys()
    first_key = next(iter(state_dict), "")
    first_model_key = next(iter(model_keys), "")

    if first_model_key.startswith("model.") and not first_key.startswith("model."):
        remapped = {f"model.{key}": value for key, value in state_dict.items()}
        model.load_state_dict(remapped)
        return

    if not first_model_key.startswith("model.") and first_key.startswith("model."):
        remapped = {key.removeprefix("model."): value for key, value in state_dict.items()}
        model.load_state_dict(remapped)
        return

    model.load_state_dict(state_dict)
