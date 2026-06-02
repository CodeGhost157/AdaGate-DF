#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from deepfake_lowres.data.manifests import balance_train_dataframe, load_celebdf_splits, load_ffpp_metadata, split_ffpp_metadata, summarize_splits
from deepfake_lowres.models.madcorn import MaDCoRN
from deepfake_lowres.models.shufflenet import ShuffleNetDeepfakeDetector
from deepfake_lowres.models.timm_binary import BinaryTimmDeepfakeDetector
from deepfake_lowres.training.torch_binary import evaluate_binary_model, evaluate_resolutions, fit_binary_model, make_loader, measure_inference_time, save_standard_outputs
from deepfake_lowres.data.datasets import FixedResolutionImageDataset
from deepfake_lowres.utils import count_parameters, get_device, load_yaml, seed_everything


def load_splits(config):
    if config["dataset"] == "ffpp":
        df = load_ffpp_metadata(config["metadata_csv"])
        train_df, val_df, test_df = split_ffpp_metadata(df)
        train_df = balance_train_dataframe(train_df, mode=config.get("balance_mode", "none"), seed=int(config.get("seed", 42)))
        return train_df, val_df, test_df
    if config["dataset"] == "celebdf":
        return load_celebdf_splits(config["celebdf_root"])
    raise ValueError(f"Unknown dataset: {config['dataset']}")


def build_model(config):
    model_name = config["model"]
    if model_name == "shufflenetv2":
        return ShuffleNetDeepfakeDetector(dropout=float(config.get("dropout", 0.2)), pretrained=True)
    if model_name == "efficientnet_b4":
        return BinaryTimmDeepfakeDetector(config.get("backbone_name", "tf_efficientnet_b4.ns_jft_in1k"), dropout=float(config.get("dropout", 0.2)), pretrained=True)
    if model_name == "madcorn":
        return MaDCoRN(base_ch=int(config.get("base_ch", 32)))
    raise ValueError(f"This script supports shufflenetv2, efficientnet_b4, and madcorn. Got: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Train a Torch baseline model.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    print(f"Device: {device}")
    train_df, val_df, test_df = load_splits(config)
    print(summarize_splits(train_df, val_df, test_df).to_string(index=False))
    model = build_model(config)
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: total={total_params:,} trainable={trainable_params:,}")
    model, history_df, train_loader, val_loader, test_loader, total_train_mins = fit_binary_model(model, train_df, val_df, test_df, config, device)
    test_results_df = evaluate_binary_model(model, test_loader, device, model_name=config["model"], uncertain_low=config.get("uncertain_low", 0.4), uncertain_high=config.get("uncertain_high", 0.6))
    res_df = evaluate_resolutions(model, test_df, config, device)
    inf_rows = []
    for res in config.get("resolutions", [config.get("image_size_for_model", 224)]):
        ds = FixedResolutionImageDataset(test_df, int(res), int(config.get("image_size_for_model", 224)))
        loader = make_loader(ds, int(config.get("batch_size", 32)), int(config.get("num_workers", 2)), train=False, device=device)
        inf_rows.append(measure_inference_time(model, loader, device, mode_name=f"source {res} -> model {config.get('image_size_for_model', 224)}").iloc[0].to_dict())
    inf_df = pd.DataFrame(inf_rows)
    train_time_df = pd.DataFrame([{"Model Type": config["model"], "mins": total_train_mins, "hrs": total_train_mins / 60.0}])
    summary_df = pd.DataFrame([{"model": config["model"], "dataset": config["dataset"], "total_params": total_params, "trainable_params": trainable_params}])
    save_standard_outputs(config.get("out_dir", "outputs"), history_df, test_results_df, res_df, inf_df, train_time_df, summary_df)
    print(test_results_df.to_string(index=False))
    print(f"Outputs saved to: {config.get('out_dir', 'outputs')}")


if __name__ == "__main__":
    main()
