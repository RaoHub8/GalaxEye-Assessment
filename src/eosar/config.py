"""Central configuration for EO-SAR training and inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    """Serializable project configuration.

    The defaults target a low-end Windows DirectML machine. CLI scripts can load
    YAML/JSON and then override selected values.
    """

    data_dir: Path = Path("dataset")
    output_dir: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 42
    deterministic: bool = True

    epochs: int = 20
    batch_size: int = 8
    accumulation_steps: int = 1
    patch_size: int = 256
    crop_size: int = 256
    stride: int = 128
    num_workers: int = 0
    prefetch_factor: int | None = None
    persistent_workers: bool = False

    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    gradient_clip: float = 1.0
    mixed_precision: bool = True

    pos_weight: float = 6.0
    focal_gamma: float = 2.0
    focal_w: float = 0.5
    dice_w: float = 0.5

    model_name: str = "lightunet"
    in_channels: int = 4
    num_classes: int = 1
    base_channels: int = 32
    channel_multipliers: tuple[int, ...] = field(default_factory=lambda: (1, 2, 4, 8))
    dropout: float = 0.05

    # Backward-compatible fields for older commands/checkpoints.
    encoder_name: str = "lightunet"
    encoder_weights: str | None = None
    encoder_depth: int = 4
    decoder_channels: tuple[int, ...] = field(default_factory=lambda: (256, 128, 64, 32))

    threshold: float = 0.5
    use_tta: bool = False
    save_predictions: bool = False

    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.0
    val_interval: int = 1
    checkpoint_interval: int = 1
    resume: bool = True
    resume_path: Path | None = None

    log_interval: int = 10
    tensorboard: bool = True
    csv_log: bool = True

    # Dataset-specific parameters (from notebook)
    positive_patch_prob: float = 0.70
    positive_coord_stride: int = 64
    max_positive_coords_per_image: int = 16
    invalid_zero_threshold: float = 2.0
    sar_log_transform: bool = False
    sar_clip_percentiles: tuple[float, float] = field(default_factory=lambda: (2.0, 98.0))
    min_blob_size: int = 24
    morph_kernel_size: int = 0
    extended_thresholds: tuple[float, ...] = field(
        default_factory=lambda: tuple([round(x, 2) for x in np_arange_like(0.20, 0.90, 0.05)] + [0.92, 0.95])
    )
    min_component_sizes: tuple[int, ...] = field(default_factory=lambda: (64, 96, 128, 256))
    cache_version: str = "v2_sar_p2p98_gaussian_invalid"
    reuse_cache: bool = True
    export_visuals: bool = True
    
    # Threshold sweep parameters
    threshold_min: float = 0.2
    threshold_max: float = 0.8
    threshold_step: float = 0.05

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        if self.resume_path is not None:
            self.resume_path = Path(self.resume_path)
        self.crop_size = int(self.patch_size or self.crop_size)
        self.patch_size = int(self.crop_size)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        data = asdict(self)
        for key in ("data_dir", "output_dir", "resume_path"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        data["channel_multipliers"] = list(self.channel_multipliers)
        data["decoder_channels"] = list(self.decoder_channels)
        data["sar_clip_percentiles"] = list(self.sar_clip_percentiles)
        data["extended_thresholds"] = list(self.extended_thresholds)
        data["min_component_sizes"] = list(self.min_component_sizes)
        return data

    def to_json(self, path: Path) -> None:
        """Save configuration as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create config from a dictionary, ignoring unknown keys."""
        valid = set(cls.__dataclass_fields__.keys())
        cleaned = {k: v for k, v in data.items() if k in valid}
        for key in ("data_dir", "output_dir", "resume_path"):
            if cleaned.get(key) is not None:
                cleaned[key] = Path(cleaned[key])
        for key in ("channel_multipliers", "decoder_channels", "sar_clip_percentiles", "extended_thresholds", "min_component_sizes"):
            if isinstance(cleaned.get(key), list):
                cleaned[key] = tuple(cleaned[key])
        return cls(**cleaned)

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        """Load configuration from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Install pyyaml to load YAML configs.") from exc
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    @classmethod
    def preset_low_memory(cls) -> "Config":
        """Stable preset for low VRAM environments."""
        return cls(
            batch_size=1,
            accumulation_steps=8,
            patch_size=256,
            stride=128,
            num_workers=0,
            base_channels=32,
            mixed_precision=True,
            encoder_weights=None,
        )

    @classmethod
    def preset_medium_memory(cls) -> "Config":
        """Preset for standard GPU setups (matches Kaggle notebook)."""
        return cls(
            batch_size=8,
            accumulation_steps=1,
            patch_size=256,
            stride=128,
            num_workers=0,
            base_channels=32,
            mixed_precision=True,
            encoder_weights=None,
        )

    @classmethod
    def preset_high_memory(cls) -> "Config":
        """Preset for CUDA GPUs such as a Kaggle T4."""
        return cls(
            batch_size=8,
            accumulation_steps=1,
            patch_size=256,
            stride=128,
            num_workers=4,
            base_channels=32,
            mixed_precision=True,
            encoder_weights=None,
        )

    @property
    def checkpoint_path(self) -> Path:
        """Best checkpoint path."""
        return self.output_dir / "best_model.pth"

    @property
    def last_checkpoint_path(self) -> Path:
        """Most recent checkpoint path."""
        return self.output_dir / "last_model.pth"

    @property
    def config_path(self) -> Path:
        """Saved config path."""
        return self.output_dir / "config.json"

    @property
    def history_path(self) -> Path:
        """JSON history path."""
        return self.output_dir / "history.json"

    @property
    def csv_path(self) -> Path:
        """CSV metrics path."""
        return self.output_dir / "metrics.csv"

    def split_root(self, split: str) -> Path:
        """Return split root for flat or nested dataset layouts."""
        outer = self.data_dir / split
        nested = outer / split
        if (nested / "pre-event").exists():
            return nested
        return outer

    @property
    def cache_dir(self) -> Path:
        """Directory for cache-backed post-training inference artifacts."""
        return self.output_dir / "cache"

    def cache_path(self, stem: str, suffix: str = ".npz") -> Path:
        """Notebook-compatible cache path keyed by preprocessing/inference version."""
        return self.cache_dir / f"{stem}_{self.cache_version}_c{self.crop_size}_s{self.stride}{suffix}"


def np_arange_like(start: float, stop: float, step: float) -> list[float]:
    """Small dependency-free helper matching ``np.arange(start, stop + eps, step)``."""
    values: list[float] = []
    cur = start
    while cur <= stop + 1e-9:
        values.append(cur)
        cur += step
    return values
