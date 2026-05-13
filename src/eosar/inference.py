"""Patch-based inference helpers for EO-SAR change detection."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from eosar.data import read_triplet


_GAUSSIAN_WINDOW_CACHE: dict[tuple[int, str, float], torch.Tensor] = {}


def _window_starts(length: int, crop: int, stride: int) -> list[int]:
    """Return sliding-window start indices that always include the image edge."""
    if length <= crop:
        return [0]
    starts = list(range(0, length - crop + 1, stride))
    last = length - crop
    if starts[-1] != last:
        starts.append(last)
    return starts


def _pad_to_crop(image_tensor: torch.Tensor, crop: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad an image tensor to at least crop size."""
    _, _, height, width = image_tensor.shape
    pad_h = max(crop - height, 0)
    pad_w = max(crop - width, 0)
    if pad_h == 0 and pad_w == 0:
        return image_tensor, (height, width)
    padded = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, (height, width)


def _gaussian_window(crop: int, device: torch.device | str | None = None, sigma_scale: float = 0.125) -> torch.Tensor:
    """Return the notebook Gaussian blending window for one crop."""
    device = torch.device("cpu") if device is None else torch.device(device)
    key = (int(crop), str(device), float(sigma_scale))
    if key not in _GAUSSIAN_WINDOW_CACHE:
        coords = torch.arange(crop, dtype=torch.float32, device=device)
        center = (crop - 1) / 2.0
        sigma = max(sigma_scale * crop, 1.0)
        gaussian_1d = torch.exp(-0.5 * ((coords - center) / sigma).pow(2))
        window = gaussian_1d[:, None] * gaussian_1d[None, :]
        _GAUSSIAN_WINDOW_CACHE[key] = window / window.max().clamp(min=1e-6)
    return _GAUSSIAN_WINDOW_CACHE[key]


def sliding_window_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    crop: int = 256,
    stride: int = 128,
    threshold: float = 0.5,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run full-image inference with Gaussian-weighted overlapping patches."""
    model.eval()
    image_tensor, original_shape = _pad_to_crop(image_tensor, crop)
    image_tensor = image_tensor.to(device, non_blocking=True)
    _, _, height, width = image_tensor.shape
    pred_sum = torch.zeros(1, 1, height, width, dtype=torch.float32, device=device)
    weight_sum = torch.zeros(1, 1, height, width, dtype=torch.float32, device=device)
    gaussian_weight = _gaussian_window(crop, device=device)

    with torch.inference_mode():
        for y in _window_starts(height, crop, stride):
            for x in _window_starts(width, crop, stride):
                patch = image_tensor[:, :, y : y + crop, x : x + crop]
                if use_amp and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = torch.sigmoid(model(patch)).float()
                else:
                    out = torch.sigmoid(model(patch)).float()
                out_h, out_w = out.shape[2], out.shape[3]
                weight = gaussian_weight[None, None, :out_h, :out_w]
                pred_sum[:, :, y : y + out_h, x : x + out_w] += out * weight
                weight_sum[:, :, y : y + out_h, x : x + out_w] += weight

    prob_map = (pred_sum / (weight_sum + 1e-6)).squeeze().detach().cpu().numpy()
    h, w = original_shape
    prob_map = prob_map[:h, :w]
    return (prob_map >= threshold).astype(np.uint8), prob_map.astype(np.float32)


def tta_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    crop: int = 256,
    stride: int = 128,
    threshold: float = 0.5,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Average original and flip test-time augmentations."""

    def run(img: torch.Tensor) -> np.ndarray:
        _, prob = sliding_window_inference(model, img, device, crop, stride, 0.0, use_amp)
        return prob

    p0 = run(image_tensor)
    p1 = np.fliplr(run(torch.flip(image_tensor, [3])))
    p2 = np.flipud(run(torch.flip(image_tensor, [2])))
    p3 = np.flipud(np.fliplr(run(torch.flip(image_tensor, [2, 3]))))
    prob = ((p0 + p1 + p2 + p3) / 4.0).astype(np.float32)
    return (prob >= threshold).astype(np.uint8), prob


def save_mask(mask: np.ndarray, reference_path: Path, output_path: Path) -> None:
    """Save a binary mask as a georeferenced uint8 TIFF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            height=int(mask.shape[0]),
            width=int(mask.shape[1]),
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)


def load_pair_tensor(pre_path: Path, post_path: Path) -> torch.Tensor:
    """Load an EO/SAR pair as a batched tensor."""
    image, _ = read_triplet(pre_path, post_path, None)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    return tensor.contiguous()


def predict_pair(
    model: torch.nn.Module,
    pre_path: Path,
    post_path: Path,
    device: torch.device,
    crop: int = 256,
    stride: int = 128,
    threshold: float = 0.5,
    use_tta: bool = False,
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a binary mask and probability map for one image pair."""
    tensor = load_pair_tensor(pre_path, post_path)
    infer = tta_inference if use_tta else sliding_window_inference
    return infer(model, tensor, device, crop, stride, threshold, use_amp)


def save_overlay(prob_map: np.ndarray, output_path: Path) -> None:
    """Save a simple probability heatmap overlay image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(prob_map, cmap="magma", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_prediction_cache(cache: list[dict], path: Path) -> None:
    """Save variable-size prediction cache entries like the notebook."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        index=np.array([item["index"] for item in cache], dtype=np.int32),
        prob=np.array([item["prob"] for item in cache], dtype=object),
        pred=np.array([item["pred"] for item in cache], dtype=object),
        gt=np.array([item["gt"] for item in cache], dtype=object),
        valid=np.array([item["valid"] for item in cache], dtype=object),
    )


def load_prediction_cache(path: Path) -> list[dict]:
    """Load a cache produced by ``save_prediction_cache``."""
    data = np.load(path, allow_pickle=True)
    return [
        {
            "index": int(index),
            "prob": np.asarray(prob, dtype=np.float32),
            "pred": np.asarray(pred, dtype=np.uint8),
            "gt": np.asarray(gt, dtype=np.uint8),
            "valid": np.asarray(valid, dtype=np.uint8),
        }
        for index, prob, pred, gt, valid in zip(
            data["index"], data["prob"], data["pred"], data["gt"], data["valid"]
        )
    ]
