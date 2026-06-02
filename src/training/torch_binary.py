from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deepfake_lowres.data.datasets import BinaryImageDataset, FixedResolutionImageDataset, MultiResolutionImageDataset
from deepfake_lowres.metrics import compute_binary_metrics
from deepfake_lowres.utils import ensure_dir


def make_loader(dataset, batch_size: int, num_workers: int, train: bool = False, device=None):
    device_type = getattr(device, "type", "cpu")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=(device_type == "cuda"),
        drop_last=train,
        persistent_workers=(num_workers > 0),
    )


def _to_binary_logits(output):
    if isinstance(output, dict):
        output = output.get("logits", output.get("fused_logits"))
    if output.ndim == 2 and output.size(1) == 2:
        return output[:, 1] - output[:, 0]
    return output.squeeze(1) if output.ndim == 2 else output


def train_one_epoch(model, loader, optimizer, scaler, device, mixed_precision=True, uncertain_low=0.4, uncertain_high=0.6):
    model.train()
    running_loss = 0.0
    y_true_all, y_prob_all = [], []
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(mixed_precision and device.type == "cuda")):
            logits = _to_binary_logits(model(x))
            loss = F.binary_cross_entropy_with_logits(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
        y_true_all.extend(labels.tolist())
        y_prob_all.extend(probs.tolist())
        running_loss += float(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    metrics = compute_binary_metrics(y_true_all, y_prob_all, uncertain_low=uncertain_low, uncertain_high=uncertain_high)
    metrics["loss"] = running_loss / max(len(loader), 1)
    return metrics


@torch.no_grad()
def validate_one_epoch(model, loader, device, desc="val", uncertain_low=0.4, uncertain_high=0.6):
    model.eval()
    running_loss = 0.0
    y_true_all, y_prob_all = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True).float()
        logits = _to_binary_logits(model(x))
        loss = F.binary_cross_entropy_with_logits(logits, y)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_true_all.extend(y.detach().cpu().numpy().tolist())
        y_prob_all.extend(probs.tolist())
        running_loss += float(loss.item())
    metrics = compute_binary_metrics(y_true_all, y_prob_all, uncertain_low=uncertain_low, uncertain_high=uncertain_high)
    metrics["loss"] = running_loss / max(len(loader), 1)
    return metrics


def fit_binary_model(model, train_df, val_df, test_df, config: dict, device):
    out_dir = ensure_dir(config.get("out_dir", "outputs"))
    ckpt_path = out_dir / config.get("ckpt_name", "best_model.pt")
    resolutions = config.get("resolutions", [128, 224, 256, 384])
    image_size_for_model = int(config.get("image_size_for_model", config.get("image_size", 224)))
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 2))
    train_mode = config.get("train_mode", "multi_resolution")

    if train_mode == "single_resolution":
        train_ds = BinaryImageDataset(train_df, image_size=image_size_for_model, train=True)
        val_ds = BinaryImageDataset(val_df, image_size=image_size_for_model, train=False)
        test_ds = BinaryImageDataset(test_df, image_size=image_size_for_model, train=False)
    else:
        train_ds = MultiResolutionImageDataset(train_df, resolutions=resolutions, image_size_for_model=image_size_for_model, mode="train_random")
        val_ds = MultiResolutionImageDataset(val_df, resolutions=resolutions, image_size_for_model=image_size_for_model, mode="eval_all")
        test_ds = MultiResolutionImageDataset(test_df, resolutions=resolutions, image_size_for_model=image_size_for_model, mode="eval_all")

    train_loader = make_loader(train_ds, batch_size, num_workers, train=True, device=device)
    val_loader = make_loader(val_ds, batch_size, num_workers, train=False, device=device)
    test_loader = make_loader(test_ds, batch_size, num_workers, train=False, device=device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 1e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    scaler = torch.cuda.amp.GradScaler(enabled=(bool(config.get("mixed_precision", True)) and device.type == "cuda"))

    history = []
    best_val_auc = -1.0
    train_start = time.time()
    epochs = int(config.get("epochs", 10))
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, config.get("mixed_precision", True), config.get("uncertain_low", 0.4), config.get("uncertain_high", 0.6))
        val_metrics = validate_one_epoch(model, val_loader, device, "val", config.get("uncertain_low", 0.4), config.get("uncertain_high", 0.6))
        epoch_mins = (time.time() - epoch_start) / 60.0
        row = {
            "Epoch": epoch,
            "Train Loss": train_metrics["loss"],
            "Train Acc": train_metrics["accuracy"],
            "Train AUC": train_metrics["auc"],
            "Val Acc": val_metrics["accuracy"],
            "Val AUC": val_metrics["auc"],
            "Precision": val_metrics["precision"],
            "Recall": val_metrics["recall"],
            "F1": val_metrics["f1"],
            "Uncertain Rate (%)": val_metrics["uncertain_rate"],
            "Avg Uncertainty": val_metrics["avg_uncertainty"],
            "Epoch mins": epoch_mins,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | Train Loss {train_metrics['loss']:.4f} | "
            f"Train Acc {train_metrics['accuracy']:.4f} | Train AUC {train_metrics['auc']:.4f} | "
            f"Val Acc {val_metrics['accuracy']:.4f} | Val AUC {val_metrics['auc']:.4f} | "
            f"F1 {val_metrics['f1']:.4f} | Uncertain {val_metrics['uncertain_rate']:.2f}% | Time {epoch_mins:.2f} min"
        )
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            torch.save({"model": model.state_dict(), "config": config, "best_val_auc": best_val_auc}, ckpt_path)
            print(f"Saved best checkpoint with Val AUC = {best_val_auc:.4f}")

    total_train_mins = (time.time() - train_start) / 60.0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    return model, pd.DataFrame(history), train_loader, val_loader, test_loader, total_train_mins


@torch.no_grad()
def evaluate_binary_model(model, loader, device, model_name="model", uncertain_low=0.4, uncertain_high=0.6):
    metrics = validate_one_epoch(model, loader, device, desc="test", uncertain_low=uncertain_low, uncertain_high=uncertain_high)
    return pd.DataFrame([{
        "Model": model_name,
        "Accuracy": metrics["accuracy"],
        "AUC": metrics["auc"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
        "Uncertain Rate (%)": metrics["uncertain_rate"],
        "Avg Uncertainty": metrics["avg_uncertainty"],
    }])


@torch.no_grad()
def evaluate_resolutions(model, test_df, config: dict, device):
    rows = []
    for res in config.get("resolutions", [128, 224, 256, 384]):
        ds = FixedResolutionImageDataset(test_df, resolution=res, image_size_for_model=int(config.get("image_size_for_model", 224)))
        loader = make_loader(ds, int(config.get("batch_size", 32)), int(config.get("num_workers", 2)), train=False, device=device)
        metrics = validate_one_epoch(model, loader, device, desc=f"res {res}", uncertain_low=config.get("uncertain_low", 0.4), uncertain_high=config.get("uncertain_high", 0.6))
        rows.append({
            "Resolution": int(res),
            "Accuracy": metrics["accuracy"],
            "AUC": metrics["auc"],
            "Fast (%)": 0.0,
            "Medium (%)": 0.0,
            "Full (%)": 100.0,
            "Avg Uncertainty": metrics["avg_uncertainty"],
            "Uncertain Rate (%)": metrics["uncertain_rate"],
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def measure_inference_time(model, loader, device, mode_name="GPU FP32", warmup=10, max_batches=50):
    model.eval()
    iterator = iter(loader)
    for _ in range(warmup):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        x = batch["image"].to(device, non_blocking=True)
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    total_images = 0
    start = time.time()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["image"].to(device, non_blocking=True)
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_images += x.size(0)
    elapsed = time.time() - start
    return pd.DataFrame([{"Mode": mode_name, "Time per Image (ms)": (elapsed / max(total_images, 1)) * 1000.0, "FPS": total_images / max(elapsed, 1e-8)}])


def save_standard_outputs(out_dir: str | Path, history_df, test_results_df, res_df, inf_df, train_time_df, summary_df=None):
    out_dir = ensure_dir(out_dir)
    history_df.to_csv(out_dir / "training_period_results.csv", index=False)
    test_results_df.to_csv(out_dir / "test_results.csv", index=False)
    res_df.to_csv(out_dir / "accuracy_based_on_resolution.csv", index=False)
    inf_df.to_csv(out_dir / "inference_time.csv", index=False)
    train_time_df.to_csv(out_dir / "training_time.csv", index=False)
    if summary_df is not None:
        summary_df.to_csv(out_dir / "experiment_summary.csv", index=False)
