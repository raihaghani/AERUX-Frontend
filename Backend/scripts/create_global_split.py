import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Create stratified global train / held-out test CSVs at SeriesInstanceUID level.

Run once before training with use_global_split=True:

    python create_global_split.py
    python create_global_split.py --test_fraction 0.2 --overwrite

Outputs (under data_splits/ by default):
    global_train_series.csv   — use for multitask + fusion training only
    global_test_series.csv    — use ONLY for infer_test_set.py / final metrics
    global_split_metadata.json

Then in config.config DataConfig set:
    use_global_split = True
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config.config import Config
from src.datasets.global_split_utils import create_global_train_test_csvs


def main():
    parser = argparse.ArgumentParser(description="Create global train/test split CSVs")
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=None,
        help="Fraction for held-out test (default: config global_test_fraction)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing global_train_series.csv / global_test_series.csv",
    )
    args = parser.parse_args()

    config = Config()
    train_path, test_path, meta = create_global_train_test_csvs(
        config,
        test_fraction=args.test_fraction,
        overwrite=args.overwrite,
    )

    print("\n" + "=" * 60)
    print("Global split written successfully")
    print("=" * 60)
    print(f"  Train: {meta['train_rows']} rows → {train_path}")
    print(f"  Test:  {meta['test_rows']} rows → {test_path}")
    print(f"  Seed:  {meta['random_seed']}  |  test_fraction: {meta['test_fraction']}")
    print("\nSet use_global_split = True in config.config.DataConfig, then train multitask → fusion.")
    print("Evaluate fusion on the held-out set with: python infer_test_set.py ...")
    print("=" * 60)


if __name__ == "__main__":
    main()
