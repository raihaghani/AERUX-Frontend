"""
Global train / held-out test split at SeriesInstanceUID level.

Used by multitask training, fusion training, and infer_test_set so that:
- Multitask K-fold and fusion training never see global test UIDs.
- Final metrics are computed only on global_test_series.csv.

Run once:  python create_global_split.py
Then set use_global_split = True in config.config.DataConfig.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from sklearn.model_selection import train_test_split


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_preprocessed_filtered_dataframe(config) -> pd.DataFrame:
    """
    Same filtering as train_fusion*.py: full train.csv → normalize modality →
    drop Unknown → keep rows with existing preprocessed .npy.
    """
    from preprocess_dataset import normalize_modality

    train_csv_path = config.data.train_csv
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"train.csv not found: {train_csv_path}")

    df = pd.read_csv(train_csv_path)
    df["Modality"] = df["Modality"].apply(normalize_modality)
    df = df[df["Modality"] != "Unknown"]

    base_dirs = {
        "CTA": config.data.preprocessed_cta_dir,
        "MRA": config.data.preprocessed_mra_dir,
        "MRI": config.data.preprocessed_mri_dir,
    }

    def _npy_exists(row):
        mod = row["Modality"]
        label_name = "aneurysm" if int(row["Aneurysm Present"]) == 1 else "no_aneurysm"
        p = os.path.join(base_dirs.get(mod, ""), label_name, f"{row['SeriesInstanceUID']}.npy")
        return os.path.exists(p)

    df = df[df.apply(_npy_exists, axis=1)].reset_index(drop=True)
    return df


def create_global_train_test_csvs(
    config,
    test_fraction: float | None = None,
    random_seed: int | None = None,
    overwrite: bool = False,
) -> tuple[str, str, dict[str, Any]]:
    """
    Stratified split on 'Aneurysm Present'. Writes CSVs and a small metadata JSON.

    Returns (path_train_csv, path_test_csv, metadata_dict).
    """
    test_fraction = (
        test_fraction
        if test_fraction is not None
        else getattr(config.data, "global_test_fraction", 0.2)
    )
    random_seed = random_seed if random_seed is not None else config.data.random_seed

    train_path = Path(config.data.global_train_split_csv)
    test_path = Path(config.data.global_test_split_csv)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = train_path.parent / "global_split_metadata.json"

    if train_path.exists() or test_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Split file(s) already exist:\n  {train_path}\n  {test_path}\n"
                "Pass overwrite=True to replace, or delete them first."
            )


    df = build_preprocessed_filtered_dataframe(config)
    if len(df) < 4:
        raise ValueError("Not enough rows after filtering to stratify split.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_fraction,
        random_state=random_seed,
        stratify=df["Aneurysm Present"],
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    meta = {
        "random_seed": random_seed,
        "test_fraction": test_fraction,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_aneurysm_positive": int(train_df["Aneurysm Present"].sum()),
        "test_aneurysm_positive": int(test_df["Aneurysm Present"].sum()),
        "train_csv": str(train_path.resolve()),
        "test_csv": str(test_path.resolve()),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return str(train_path), str(test_path), meta


def require_global_split_files(config) -> None:
    """Raise FileNotFoundError with instructions if global split is enabled but files missing."""
    if not getattr(config.data, "use_global_split", False):
        return
    train_p = getattr(config.data, "global_train_split_csv", "")
    test_p = getattr(config.data, "global_test_split_csv", "")
    missing = [p for p in (train_p, test_p) if p and not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "use_global_split is True but split file(s) are missing:\n  "
            + "\n  ".join(missing)
            + "\n\nCreate them with:\n  python create_global_split.py"
        )


def load_multitask_source_dataframe(config) -> pd.DataFrame:
    """
    DataFrame for multitask training: global train only if use_global_split, else full train.csv.
    """
    require_global_split_files(config)

    if getattr(config.data, "use_global_split", False):
        path = config.data.global_train_split_csv
        df = pd.read_csv(path)
        print(f"Loaded GLOBAL TRAIN split ({len(df)} series) — held-out test excluded.")
        return df

    df = pd.read_csv(config.data.train_csv)
    print(f"Loaded full dataset: {len(df)} series")
    return df


def load_fusion_training_dataframes(config, train_csv_path: str):
    """
    Build fusion train/val (+ optional legacy test) DataFrames.

    Returns:
      train_df, val_df, test_df_or_none, split_mode
      split_mode is 'global' or 'legacy'
    """
    require_global_split_files(config)

    from preprocess_dataset import normalize_modality

    if getattr(config.data, "use_global_split", False):
        if not os.path.exists(config.data.global_train_split_csv):
            raise FileNotFoundError(config.data.global_train_split_csv)

        df = pd.read_csv(config.data.global_train_split_csv)
        df["Modality"] = df["Modality"].apply(normalize_modality)
        val_frac = float(getattr(config.data, "fusion_val_fraction", 0.15))
        train_df, val_df = train_test_split(
            df,
            test_size=val_frac,
            random_state=config.data.random_seed,
            stratify=df["Aneurysm Present"],
        )
        print("\n[Global split] Fusion uses GLOBAL TRAIN only; stratified train/val for fusion.")
        print(f"  Fusion train: {len(train_df)}  |  Fusion val: {len(val_df)}")
        print("  Final unbiased evaluation: run infer_test_set.py on global_test_series.csv")
        return train_df, val_df, None, "global"

    # Legacy: full filtered pool → 70% / 15% / 15%
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(train_csv_path)

    df = pd.read_csv(train_csv_path)
    df["Modality"] = df["Modality"].apply(normalize_modality)
    df = df[df["Modality"] != "Unknown"]

    base_dirs_check = {
        "CTA": config.data.preprocessed_cta_dir,
        "MRA": config.data.preprocessed_mra_dir,
        "MRI": config.data.preprocessed_mri_dir,
    }

    def _npy_exists(row):
        mod = row["Modality"]
        label_name = "aneurysm" if int(row["Aneurysm Present"]) == 1 else "no_aneurysm"
        path = os.path.join(base_dirs_check.get(mod, ""), label_name, f"{row['SeriesInstanceUID']}.npy")
        return os.path.exists(path)

    df = df[df.apply(_npy_exists, axis=1)].reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=config.data.random_seed, stratify=df["Aneurysm Present"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=config.data.random_seed, stratify=temp_df["Aneurysm Present"]
    )
    print("\n[Legacy split] 70% train / 15% val / 15% test (internal fusion test — not leak-free vs multitask).")
    print(f"  Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
    return train_df, val_df, test_df, "legacy"


def load_global_test_dataframe(config) -> pd.DataFrame:
    """Held-out test rows for infer_test_set.py."""
    if not getattr(config.data, "use_global_split", False):
        raise ValueError("load_global_test_dataframe requires config.data.use_global_split True")
    require_global_split_files(config)
    path = config.data.global_test_split_csv
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Global test CSV not found: {path}\nRun: python create_global_split.py"
        )
    return pd.read_csv(path)


def load_legacy_internal_fusion_test_dataframe(config) -> pd.DataFrame:
    """
    Legacy internal fusion test split (15% of filtered pool, stratified).
    Same rule as historical train_fusion 70/15/15 test portion.
    """
    from preprocess_dataset import normalize_modality

    df = pd.read_csv(config.data.train_csv)
    df["Modality"] = df["Modality"].apply(normalize_modality)
    df = df[df["Modality"] != "Unknown"]

    base_dirs_check = {
        "CTA": config.data.preprocessed_cta_dir,
        "MRA": config.data.preprocessed_mra_dir,
        "MRI": config.data.preprocessed_mri_dir,
    }

    def _npy_exists(row):
        mod = row["Modality"]
        label_name = "aneurysm" if int(row["Aneurysm Present"]) == 1 else "no_aneurysm"
        path = os.path.join(base_dirs_check.get(mod, ""), label_name, f"{row['SeriesInstanceUID']}.npy")
        return os.path.exists(path)

    df = df[df.apply(_npy_exists, axis=1)].reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=config.data.random_seed, stratify=df["Aneurysm Present"]
    )
    _, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=config.data.random_seed, stratify=temp_df["Aneurysm Present"]
    )
    return test_df.reset_index(drop=True)


def load_evaluation_test_dataframe(config) -> pd.DataFrame:
    """Held-out global test (recommended) or legacy internal fusion test."""
    if getattr(config.data, "use_global_split", False):
        return load_global_test_dataframe(config)
    return load_legacy_internal_fusion_test_dataframe(config)
