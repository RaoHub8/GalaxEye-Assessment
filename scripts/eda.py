"""Run the Kaggle notebook EDA cells from the structured repository."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eosar.config import Config
from eosar.eda import run_eda
from eosar.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run notebook-faithful EO-SAR EDA")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--train-limit", type=int, default=80)
    parser.add_argument("--eval-limit", type=int, default=60)
    parser.add_argument("--examples-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(data_dir=args.data_dir, output_dir=args.output_dir, seed=args.seed)
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    run_eda(
        cfg,
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
        examples_split=args.examples_split,
        examples=args.examples,
    )


if __name__ == "__main__":
    main()
