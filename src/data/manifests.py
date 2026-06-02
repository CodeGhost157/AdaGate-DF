from __future__ import annotations

from pathlib import Path
import random

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_rows(folder: Path, label: int, label_name: str):
    rows = []
    if not folder.exists():
        return rows
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rows.append({"image_path": str(path), "label": int(label), "label_name": label_name})
    return rows


def build_csv_from_binary_split(split_path: str | Path, real_name: str = "real", fake_name: str = "fake") -> pd.DataFrame:
    split_path = Path(split_path)
    rows = []
    rows.extend(_image_rows(split_path / real_name, 0, "Real"))
    rows.extend(_image_rows(split_path / fake_name, 1, "Fake"))
    return pd.DataFrame(rows)


def load_celebdf_splits(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    train_df = build_csv_from_binary_split(root / "Train")
    val_df = build_csv_from_binary_split(root / "Val")
    test_df = build_csv_from_binary_split(root / "Test")
    return train_df, val_df, test_df


def load_ffpp_metadata(metadata_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    required = {"frame_path", "split", "label", "label_name", "video_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {missing}")
    df = df[df["frame_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    df = df.rename(columns={"frame_path": "image_path"})
    df["is_oversampled_real"] = False
    return df


def split_ffpp_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["split"].str.lower() == "train"].reset_index(drop=True)
    val_df = df[df["split"].str.lower() == "val"].reset_index(drop=True)
    test_df = df[df["split"].str.lower() == "test"].reset_index(drop=True)
    return train_df, val_df, test_df


def balance_train_dataframe(train_df: pd.DataFrame, mode: str = "none", seed: int = 42) -> pd.DataFrame:
    train_df = train_df.copy().reset_index(drop=True)
    train_df["is_oversampled_real"] = False
    if mode in [None, "none"]:
        return train_df

    real_df = train_df[train_df["label"] == 0].copy()
    fake_df = train_df[train_df["label"] == 1].copy()
    if len(real_df) == 0 or len(fake_df) == 0:
        raise ValueError("Both classes must be present before balancing.")

    if mode == "undersample":
        n = min(len(real_df), len(fake_df))
        out = pd.concat([
            real_df.sample(n=n, random_state=seed),
            fake_df.sample(n=n, random_state=seed),
        ])
        out["is_oversampled_real"] = False
        return out.sample(frac=1, random_state=seed).reset_index(drop=True)

    if mode == "oversample_real":
        real_os = real_df.sample(n=len(fake_df), replace=True, random_state=seed).copy()
        real_os["is_oversampled_real"] = True
        fake_df["is_oversampled_real"] = False
        return pd.concat([real_os, fake_df]).sample(frac=1, random_state=seed).reset_index(drop=True)

    if mode == "oversample_minority":
        if len(real_df) < len(fake_df):
            minority, majority = real_df, fake_df
        else:
            minority, majority = fake_df, real_df
        minority_os = minority.sample(n=len(majority), replace=True, random_state=seed).copy()
        return pd.concat([minority_os, majority]).sample(frac=1, random_state=seed).reset_index(drop=True)

    raise ValueError(f"Unknown balance mode: {mode}")


def summarize_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rows.append({
            "split": split_name,
            "real": int((df["label"] == 0).sum()),
            "fake": int((df["label"] == 1).sum()),
            "total": int(len(df)),
        })
    return pd.DataFrame(rows)


def save_split_manifests(train_df, val_df, test_df, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
