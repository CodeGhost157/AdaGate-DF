#!/usr/bin/env python3
from __future__ import annotations

import argparse

from deepfake_lowres.data.manifests import load_celebdf_splits, save_split_manifests, summarize_splits


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test CSV manifests for Celeb-DF image folders.")
    parser.add_argument("--root", required=True, help="Path to Celeb_V2 with Train/Val/Test folders.")
    parser.add_argument("--out-dir", required=True, help="Folder where train.csv, val.csv, and test.csv will be saved.")
    args = parser.parse_args()
    train_df, val_df, test_df = load_celebdf_splits(args.root)
    save_split_manifests(train_df, val_df, test_df, args.out_dir)
    print(summarize_splits(train_df, val_df, test_df).to_string(index=False))


if __name__ == "__main__":
    main()
