"""Dataset and dataloader utilities for EO-SAR TIFF imagery matching Kaggle notebook."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from eosar.config import Config

try:
    import cv2
except Exception:
    cv2 = None

logger = logging.getLogger("eosar")

# Will be populated at module load time with statistics from training split
MODALITY_STATS = {
    "eo_mean": np.array([0.1, 0.1, 0.1], dtype=np.float32),
    "eo_std": np.array([1.0, 1.0, 1.0], dtype=np.float32),
}


def set_modality_stats(eo_mean: np.ndarray, eo_std: np.ndarray) -> None:
    """Update global modality statistics (call after EDA)."""
    global MODALITY_STATS
    MODALITY_STATS = {
        "eo_mean": np.asarray(eo_mean, dtype=np.float32),
        "eo_std": np.maximum(np.asarray(eo_std, dtype=np.float32), 1e-6),
    }


def compute_modality_stats(cfg: Config, split: str = "train", limit: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Compute and set EO mean/std statistics from a dataset split.
    
    This should be called before training to normalize EO data consistently.
    """
    split_root = cfg.data_dir / split / split
    if not split_root.exists():
        logger.warning("Split %s not found, using default MODALITY_STATS", split)
        return MODALITY_STATS["eo_mean"], MODALITY_STATS["eo_std"]
    
    samples = list_samples(split_root)
    if not samples:
        logger.warning("No samples found in %s, using default MODALITY_STATS", split)
        return MODALITY_STATS["eo_mean"], MODALITY_STATS["eo_std"]
    
    rng = np.random.default_rng(cfg.seed)
    indices = sample_indices(len(samples), limit, cfg.seed) if len(samples) > limit else np.arange(len(samples))
    
    eo_sum = np.zeros(3, dtype=np.float64)
    eo_sumsq = np.zeros(3, dtype=np.float64)
    eo_count = 0
    
    from tqdm import tqdm
    
    for i in tqdm(indices[:limit], desc="Computing modality statistics", leave=False):
        pre_path, post_path, mask_path = samples[int(i)]
        
        with rasterio.open(pre_path) as src:
            eo_raw = src.read().astype(np.float32)
        
        with rasterio.open(post_path) as src:
            sar_raw = src.read().astype(np.float32)
        
        # Compute validity mask to exclude borders
        valid = raw_validity_mask(eo_raw, sar_raw, cfg.invalid_zero_threshold)
        valid_bool = valid > 0
        
        # Process EO
        eo = eo_raw[:3].astype(np.float32)
        if eo.shape[0] == 1:
            eo = np.repeat(eo, 3, axis=0)
        if eo.max(initial=0) > 1.5:
            eo = eo / 255.0
        
        eo_valid = eo[:, valid_bool]
        eo_sum += eo_valid.sum(axis=1)
        eo_sumsq += (eo_valid ** 2).sum(axis=1)
        eo_count += eo_valid.shape[1]
    
    if eo_count > 0:
        eo_mean = eo_sum / eo_count
        eo_std = np.sqrt(np.maximum(eo_sumsq / eo_count - eo_mean ** 2, 1e-8))
    else:
        eo_mean = np.array([0.1, 0.1, 0.1], dtype=np.float32)
        eo_std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    
    set_modality_stats(eo_mean, eo_std)
    logger.info(f"Computed EO statistics: mean={eo_mean}, std={eo_std}")
    return MODALITY_STATS["eo_mean"], MODALITY_STATS["eo_std"]


def sample_indices(n: int, limit: int = 80, seed: int = 42) -> np.ndarray:
    """Return indices for sampling, respecting limit."""
    rng = np.random.default_rng(seed)
    if n <= limit:
        return np.arange(n)
    return np.sort(rng.choice(n, size=limit, replace=False))


def _ensure_chw(array: np.ndarray) -> np.ndarray:
    """Ensure a raster is channel-first."""
    if array.ndim == 2:
        return array[None, ...]
    return array


def raw_validity_mask(
    eo_raw: np.ndarray,
    sar_raw: np.ndarray,
    zero_threshold: float | None = None,
) -> np.ndarray:
    """Compute validity mask excluding co-registration black borders.
    
    Returns float32 mask where 1.0 = valid pixel, 0.0 = invalid (EO or SAR near-zero).
    """
    if zero_threshold is None:
        zero_threshold = 2.0  # default from notebook Config
    
    eo = eo_raw[:3].astype(np.float32)
    if eo.shape[0] == 1:
        eo = np.repeat(eo, 3, axis=0)
    
    sar = sar_raw[:1].astype(np.float32)
    
    # Both EO and SAR must be above threshold to be valid
    invalid = (np.max(np.abs(eo), axis=0) <= zero_threshold) & (
        np.max(np.abs(sar), axis=0) <= zero_threshold
    )
    return (~invalid).astype(np.float32)


def preprocess_eo(
    eo: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize EO channels with z-score normalization using training statistics."""
    eo = _ensure_chw(eo).astype(np.float32)
    if eo.shape[0] > 3:
        eo = eo[:3]
    if eo.shape[0] == 1:
        eo = np.repeat(eo, 3, axis=0)
    
    # Convert to [0, 1] if in [0, 255]
    if eo.max(initial=0) > 1.5:
        eo = eo / 255.0
    eo = np.clip(eo, 0.0, 1.0)
    
    # Z-score normalization with modality statistics
    mean = MODALITY_STATS["eo_mean"] if mean is None else np.asarray(mean, dtype=np.float32)
    std = MODALITY_STATS["eo_std"] if std is None else np.asarray(std, dtype=np.float32)
    
    return ((eo - mean[:, None, None]) / std[:, None, None]).astype(np.float32)


def preprocess_sar(
    sar: np.ndarray,
    log_transform: bool | None = None,
    clip_percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """Normalize SAR backscatter with per-image percentile clipping.

    The final notebook clips each SAR image to p2/p98 and min-max normalizes
    that clipped range to [0, 1]. ``clip_percentiles`` remains configurable for
    backward-compatible experiments, but the repository default is p2/p98.
    """
    sar = _ensure_chw(sar).astype(np.float32)[:1]
    sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0)

    use_log = bool(log_transform) if log_transform is not None else False
    if use_log:
        sar = np.log1p(np.maximum(sar, 0.0))

    if clip_percentiles is None:
        clip_percentiles = (2.0, 98.0)

    p_low, p_high = np.percentile(sar, clip_percentiles)
    sar = np.clip(sar, p_low, p_high)
    sar = (sar - p_low) / (p_high - p_low + 1e-6)
    sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0)
    return sar.astype(np.float32)


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Convert binary or damage-level masks into {0, 1} float masks."""
    mask = np.asarray(mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[0]
    
    if mask.max(initial=0) > 1:
        mask = np.where(mask >= 2, 1.0, 0.0)
    else:
        mask = np.where(mask > 0, 1.0, 0.0)
    
    return mask.astype(np.float32)


def pad_chw_or_hw(
    arr: np.ndarray,
    min_h: int,
    min_w: int,
    value: float = 0.0,
) -> np.ndarray:
    """Pad array to at least (min_h, min_w), preserving channel dimension if present."""
    if arr.ndim == 3:
        c, h, w = arr.shape
        out = np.full((c, max(h, min_h), max(w, min_w)), value, dtype=arr.dtype)
        out[:, :h, :w] = arr
    else:
        h, w = arr.shape
        out = np.full((max(h, min_h), max(w, min_w)), value, dtype=arr.dtype)
        out[:h, :w] = arr
    return out


def list_samples(root: Path) -> list[tuple[Path, Path, Path]]:
    """List all matched (pre-event, post-event, target) triplets in a split."""
    pre_dir = root / "pre-event"
    post_dir = root / "post-event"
    target_dir = root / "target"
    
    samples = []
    for pre_path in sorted(pre_dir.glob("*.tif")):
        post_path = post_dir / pre_path.name
        mask_path = target_dir / pre_path.name
        if post_path.exists() and mask_path.exists():
            samples.append((pre_path, post_path, mask_path))
    
    return samples


def read_mask(mask_path: Path) -> np.ndarray:
    """Read mask and convert to binary {0, 1}."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.float32)
    return preprocess_mask(mask)



# ============================================================================
# Custom Augmentations (matching notebook)
# ============================================================================


class EOColorJitter(A.ImageOnlyTransform):
    """Modality-specific: jitter EO RGB only."""

    def __init__(
        self,
        brightness: float = 0.08,
        contrast: float = 0.10,
        always_apply: bool = False,
        p: float = 0.35,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.brightness = brightness
        self.contrast = contrast

    def get_params(self) -> dict:
        return {
            "b": random.uniform(-self.brightness, self.brightness),
            "c": random.uniform(1.0 - self.contrast, 1.0 + self.contrast),
        }

    def apply(self, image: np.ndarray, b: float = 0.0, c: float = 1.0, **params) -> np.ndarray:
        out = image.copy()
        out[..., :3] = out[..., :3] * c + b
        return out


class EOMildGaussianBlur(A.ImageOnlyTransform):
    """Modality-specific: blur EO RGB only."""

    def __init__(self, always_apply: bool = False, p: float = 0.15):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        if cv2 is None:
            return image
        out = image.copy()
        out[..., :3] = cv2.GaussianBlur(out[..., :3], (3, 3), 0)
        return out


class SARSpeckleAndScale(A.ImageOnlyTransform):
    """Modality-specific: add speckle noise to SAR channel."""

    def __init__(
        self,
        noise_std: float = 0.08,
        scale_range: tuple[float, float] = (0.90, 1.10),
        always_apply: bool = False,
        p: float = 0.35,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.noise_std = noise_std
        self.scale_range = scale_range

    def get_params(self) -> dict:
        return {"scale": random.uniform(*self.scale_range)}

    def apply(self, image: np.ndarray, scale: float = 1.0, **params) -> np.ndarray:
        out = image.copy()
        sar = out[..., 3:4]
        noise = np.random.normal(loc=1.0, scale=self.noise_std, size=sar.shape).astype(np.float32)
        out[..., 3:4] = sar * noise * scale
        return out


def read_triplet(
    pre_path: Path,
    post_path: Path,
    mask_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read and preprocess one EO/SAR/mask triplet."""
    with rasterio.open(pre_path) as src:
        eo = preprocess_eo(src.read())
    with rasterio.open(post_path) as src:
        sar = preprocess_sar(src.read())

    if eo.shape[1:] != sar.shape[1:]:
        raise ValueError(f"Shape mismatch: {pre_path.name} EO {eo.shape} SAR {sar.shape}")

    image = np.concatenate([eo, sar], axis=0)
    image = np.transpose(image, (1, 2, 0))

    mask = None
    if mask_path is not None:
        with rasterio.open(mask_path) as src:
            mask = preprocess_mask(src.read(1))
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"Mask shape mismatch: {mask_path.name} mask {mask.shape} image {image.shape[:2]}"
            )
    return image, mask



# ============================================================================
# Dataset
# ============================================================================


class EOSARDataset(Dataset):
    """EO/SAR change detection dataset with change-aware crop sampling and validity masks."""

    def __init__(
        self,
        split: str = "train",
        transform: A.Compose | None = None,
        change_aware: bool = False,
        cfg: Config | None = None,
    ):
        """
        Args:
            split: "train", "val", or "test"
            transform: albumentations Compose
            change_aware: if True, bias 70% of crops toward positive patches
            cfg: Config object (if None, uses defaults)
        """
        if cfg is None:
            cfg = Config()
        
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.change_aware = bool(change_aware)
        
        # RNG seeded per split for reproducibility
        self.rng = np.random.default_rng(
            cfg.seed + {"train": 0, "val": 10, "test": 20}.get(split, 0)
        )
        
        # Find the split directory
        split_root = cfg.data_dir / split / split
        if not split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {split_root}")
        
        self.root = split_root
        self.samples = list_samples(self.root)
        
        if not self.samples:
            raise RuntimeError(f"No matched triplets found in {self.root}")
        
        logger.info(f"[{split:5s}] {len(self.samples)} matched triplets loaded")
        
        # Precompute change-aware positive coordinates if enabled
        self.positive_coords = self._precompute_positive_coords() if self.change_aware else {}

    def _precompute_positive_coords(self) -> dict[int, list[tuple[int, int]]]:
        """Cache positive crop coordinates for each image (if any)."""
        coords = {}
        crop = self.cfg.crop_size
        stride = self.cfg.positive_coord_stride
        max_per = self.cfg.max_positive_coords_per_image
        
        from tqdm import tqdm
        
        for idx, (_, _, mask_path) in enumerate(
            tqdm(self.samples, desc="Cache positive crop coords", leave=False)
        ):
            mask = read_mask(mask_path)
            h, w = mask.shape
            
            # Generate candidate positions
            ys = list(range(0, max(h - crop + 1, 1), stride))
            xs = list(range(0, max(w - crop + 1, 1), stride))
            
            # Ensure we cover the image edges
            if not ys or ys[-1] != max(h - crop, 0):
                ys.append(max(h - crop, 0))
            if not xs or xs[-1] != max(w - crop, 0):
                xs.append(max(w - crop, 0))
            
            # Find which positions have positive pixels
            found = []
            for y in ys:
                for x in xs:
                    if mask[y : y + crop, x : x + crop].sum() > 0:
                        found.append((int(y), int(x)))
            
            # Limit to max_per
            if len(found) > max_per:
                keep = self.rng.choice(len(found), size=max_per, replace=False)
                found = [found[int(k)] for k in keep]
            
            if found:
                coords[idx] = found
        
        logger.info(
            f"  positive crop coordinates cached for {len(coords)}/{len(self.samples)} images"
        )
        return coords

    def __len__(self) -> int:
        return len(self.samples)

    def _choose_crop(self, idx: int, h: int, w: int) -> tuple[int, int, int]:
        """Choose a crop location, matching the notebook's second positive-bias check."""
        crop = self.cfg.crop_size
        if (
            self.change_aware
            and self.positive_coords
            and self.rng.random() < self.cfg.positive_patch_prob
        ):
            positive_image_ids = list(self.positive_coords.keys())
            idx = int(self.rng.choice(positive_image_ids))
            choices = self.positive_coords[idx]
            y, x = choices[int(self.rng.integers(0, len(choices)))]
            return idx, int(y), int(x)
        y = int(self.rng.integers(0, max(h - crop + 1, 1)))
        x = int(self.rng.integers(0, max(w - crop + 1, 1)))
        return idx, y, x

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (image, mask, valid_mask) for one sample."""
        # For training: optionally force a positive patch
        forced_crop = None
        if (
            self.split == "train"
            and self.change_aware
            and self.positive_coords
            and self.rng.random() < self.cfg.positive_patch_prob
        ):
            positive_image_ids = list(self.positive_coords.keys())
            idx = int(self.rng.choice(positive_image_ids))
            choices = self.positive_coords[idx]
            forced_crop = choices[int(self.rng.integers(0, len(choices)))]

        pre_path, post_path, mask_path = self.samples[idx]

        # Read raw data
        with rasterio.open(pre_path) as src:
            eo_raw = src.read().astype(np.float32)
        with rasterio.open(post_path) as src:
            sar_raw = src.read().astype(np.float32)
        with rasterio.open(mask_path) as src:
            mask_raw = src.read(1).astype(np.float32)

        # Compute validity mask
        valid = raw_validity_mask(eo_raw, sar_raw, self.cfg.invalid_zero_threshold)

        # Preprocess
        eo = preprocess_eo(eo_raw)
        sar = preprocess_sar(
            sar_raw,
            log_transform=self.cfg.sar_log_transform,
            clip_percentiles=self.cfg.sar_clip_percentiles,
        )
        mask = preprocess_mask(mask_raw)

        # Training: crop to fixed size with optional forced positive location
        if self.split == "train":
            h, w = mask.shape
            
            # Pad if image is smaller than crop size
            if h < self.cfg.crop_size or w < self.cfg.crop_size:
                eo = pad_chw_or_hw(eo, self.cfg.crop_size, self.cfg.crop_size)
                sar = pad_chw_or_hw(sar, self.cfg.crop_size, self.cfg.crop_size)
                mask = pad_chw_or_hw(mask, self.cfg.crop_size, self.cfg.crop_size)
                valid = pad_chw_or_hw(valid, self.cfg.crop_size, self.cfg.crop_size)
                h, w = mask.shape
            
            # Choose crop location
            if forced_crop is not None:
                y, x = forced_crop
            else:
                _, y, x = self._choose_crop(idx, h, w)
            
            # Extract crop
            crop = self.cfg.crop_size
            eo = eo[:, y : y + crop, x : x + crop]
            sar = sar[:, y : y + crop, x : x + crop]
            mask = mask[y : y + crop, x : x + crop]
            valid = valid[y : y + crop, x : x + crop]

        # Stack EO and SAR, convert to HWC for augmentation
        image = np.concatenate([eo, sar], axis=0).transpose(1, 2, 0)

        # Apply augmentation if provided
        if self.transform:
            aug = self.transform(image=image, mask=mask, valid_mask=valid)
            image = aug["image"].float()
            mask = aug["mask"]
            valid = aug["valid_mask"]
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask)
            if not isinstance(valid, torch.Tensor):
                valid = torch.as_tensor(valid)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask)
            valid = torch.from_numpy(valid)

        return (
            image.contiguous(),
            mask.unsqueeze(0).float().contiguous(),
            valid.unsqueeze(0).float().contiguous(),
        )



# ============================================================================
# Augmentations (matching notebook)
# ============================================================================


def build_train_transform(cfg: Config) -> A.Compose:
    """Build training augmentation pipeline matching notebook."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            EOColorJitter(brightness=0.08, contrast=0.10, p=0.35),
            EOMildGaussianBlur(p=0.15),
            SARSpeckleAndScale(noise_std=0.08, scale_range=(0.90, 1.10), p=0.35),
            ToTensorV2(),
        ],
        additional_targets={"valid_mask": "mask"},
    )


def build_inference_transform(cfg: Config | None = None) -> A.Compose:
    """Build inference (val/test) augmentation pipeline."""
    return A.Compose(
        [ToTensorV2()],
        additional_targets={"valid_mask": "mask"},
    )


def build_train_transform_legacy(patch_size: int) -> A.Compose:
    """Legacy transform for backward compatibility."""
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=patch_size,
                min_width=patch_size,
                border_mode=0,
                value=0,
                mask_value=0,
            ),
            A.RandomCrop(patch_size, patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2(),
        ]
    )


def build_inference_transform_legacy() -> A.Compose:
    """Legacy inference transform for backward compatibility."""
    return A.Compose([ToTensorV2()])


def _loader_kwargs(cfg: Config, shuffle: bool, batch_size: int) -> dict:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
        "pin_memory": False,
        "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
    }
    if cfg.num_workers > 0 and cfg.prefetch_factor is not None:
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return kwargs


# ============================================================================
# Dataloaders
# ============================================================================


def build_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, test dataloaders matching notebook."""
    train_ds = EOSARDataset(
        split="train",
        transform=build_train_transform(cfg),
        change_aware=True,
        cfg=cfg,
    )
    val_ds = EOSARDataset(
        split="val",
        transform=build_inference_transform(cfg),
        change_aware=False,
        cfg=cfg,
    )
    test_ds = EOSARDataset(
        split="test",
        transform=build_inference_transform(cfg),
        change_aware=False,
        cfg=cfg,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
