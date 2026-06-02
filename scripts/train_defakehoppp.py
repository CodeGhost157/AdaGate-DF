#!/usr/bin/env python3
from __future__ import annotations

import argparse
import joblib
import pandas as pd

from deepfake_lowres.data.manifests import balance_train_dataframe, load_ffpp_metadata, split_ffpp_metadata, summarize_splits
from deepfake_lowres.models.defakehoppp import DefakeHopPPConfig, evaluate_model, evaluate_resolutions, fit_defakehoppp_pipeline
from deepfake_lowres.utils import ensure_dir, load_yaml, seed_everything


def main():
    parser = argparse.ArgumentParser(description="Train the DeFakeHop++ approximate classical baseline.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config.get("seed", 42)))
    df = load_ffpp_metadata(config["metadata_csv"])
    train_df, val_df, test_df = split_ffpp_metadata(df)
    train_df = balance_train_dataframe(train_df, mode=config.get("balance_mode", "none"), seed=int(config.get("seed", 42)))
    print(summarize_splits(train_df, val_df, test_df).to_string(index=False))
    cfg = DefakeHopPPConfig(
        image_size=int(config.get("image_size", 224)),
        resolutions=tuple(config.get("resolutions", [128, 224, 256, 384])),
        small_block_size=int(config.get("small_block_size", 13)),
        large_block_size=int(config.get("large_block_size", 32)),
        patch_size=int(config.get("patch_size", 3)),
        stride=int(config.get("stride", 2)),
        max_train_fit_per_class=int(config.get("max_train_fit_per_class", 1500)),
        spatial_pca_energy=float(config.get("spatial_pca_energy", 0.8)),
        spatial_pca_max_channels=int(config.get("spatial_pca_max_channels", 10)),
        lgb_num_leaves=int(config.get("lgb_num_leaves", 64)),
        lgb_n_estimators=int(config.get("lgb_n_estimators", 300)),
        lgb_learning_rate=float(config.get("lgb_learning_rate", 0.05)),
        lgb_subsample=float(config.get("lgb_subsample", 0.8)),
        lgb_colsample_bytree=float(config.get("lgb_colsample_bytree", 0.8)),
        uncertain_low=float(config.get("uncertain_low", 0.4)),
        uncertain_high=float(config.get("uncertain_high", 0.6)),
        random_state=int(config.get("seed", 42)),
    )
    model = fit_defakehoppp_pipeline(train_df, val_df, cfg)
    out_dir = ensure_dir(config.get("out_dir", "outputs/defakehoppp"))
    joblib.dump(model, out_dir / "defakehoppp_model.joblib")
    test_results_df = evaluate_model(model, test_df)
    res_df = evaluate_resolutions(model, test_df, cfg.resolutions)
    train_time_df = pd.DataFrame([{"Model Type": "DeFakeHop++ Approx", "mins": model["training_time_minutes"], "hrs": model["training_time_minutes"] / 60.0}])
    test_results_df.to_csv(out_dir / "test_results.csv", index=False)
    res_df.to_csv(out_dir / "accuracy_based_on_resolution.csv", index=False)
    train_time_df.to_csv(out_dir / "training_time.csv", index=False)
    print(test_results_df.to_string(index=False))
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
