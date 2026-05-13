import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eosar.config import Config


def count_tifs(root: Path) -> tuple[int, int, int]:
    return (
        len(list((root / "pre-event").glob("*.tif"))),
        len(list((root / "post-event").glob("*.tif"))),
        len(list((root / "target").glob("*.tif"))),
    )


def label_stats(root: Path, limit: int = 5) -> tuple[list[float], float]:
    values = set()
    positive_ratios = []
    for mask_path in sorted((root / "target").glob("*.tif"))[:limit]:
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)
        values.update(float(v) for v in np.unique(mask))
        binary = np.where(mask >= 2, 1.0, 0.0) if mask.max(initial=0) > 1 else np.where(mask > 0, 1.0, 0.0)
        positive_ratios.append(float(binary.mean()))
    return sorted(values), float(np.mean(positive_ratios)) if positive_ratios else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    cfg = Config(data_dir=args.data_dir)
    expected = {
        "train": (2781, 2781, 2781),
        "val": (334, 334, 334),
        "test": (77, 77, 77),
    }

    ok = True
    print(f"{'Split':<8}{'Pre':>8}{'Post':>8}{'Target':>8}{'Mean+':>10}  Label values sample  Root")
    for split in ("train", "val", "test"):
        root = cfg.split_root(split)
        counts = count_tifs(root)
        values, pos_ratio = label_stats(root)
        ok = ok and counts == expected[split]
        print(f"{split:<8}{counts[0]:>8}{counts[1]:>8}{counts[2]:>8}{100 * pos_ratio:>9.3f}%  {values}  {root}")

    if not ok:
        raise SystemExit("Dataset verification failed.")

    print("Dataset verification passed.")


if __name__ == "__main__":
    main()
