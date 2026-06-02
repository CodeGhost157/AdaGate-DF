#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd

from deepfake_lowres.data.manifests import balance_train_dataframe, load_ffpp_metadata, split_ffpp_metadata, summarize_splits
from deepfake_lowres.training.gated_runner import evaluate_gated_model, fit_gated_model
from deepfake_lowres.utils import count_parameters, get_device, load_yaml, save_json, seed_everything


def main():
    parser = argparse.ArgumentParser(description="Train the gated dual-branch deepfake detector.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    df = load_ffpp_metadata(config["metadata_csv"])
    train_df, val_df, test_df = split_ffpp_metadata(df)
    train_df = balance_train_dataframe(train_df, mode=config.get("balance_mode", "none"), seed=int(config.get("seed", 42)))
    print(summarize_splits(train_df, val_df, test_df).to_string(index=False))
    model, history_df, train_loader, val_loader, test_loader, total_train_mins = fit_gated_model(train_df, val_df, test_df, config, device)
    total_params, trainable_params = count_parameters(model)
    test_metrics = evaluate_gated_model(model, test_loader, device, threshold=float(config.get("uncertain_threshold", 0.65)))
    out_dir = config.get("out_dir", "outputs/gated_dual_branch")
    import os
    os.makedirs(out_dir, exist_ok=True)
    history_df.to_csv(f"{out_dir}/training_period_results.csv", index=False)
    save_json(test_metrics, f"{out_dir}/test_metrics.json")
    summary = {
        "model": "gated_dual_branch",
        "dataset": config.get("dataset", "ffpp"),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "training_minutes": total_train_mins,
        "test": test_metrics,
    }
    save_json(summary, f"{out_dir}/summary.json")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f} AUC: {test_metrics['auc']:.4f}")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
