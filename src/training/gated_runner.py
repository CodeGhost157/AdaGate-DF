from __future__ import annotations

import time
from collections import Counter

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deepfake_lowres.data.gated_dataset import GatedMultiResolutionDataset, make_gate_targets_from_quality_features
from deepfake_lowres.metrics import compute_multiclass_binary_metrics
from deepfake_lowres.models.gated_dual_branch import GatedDualBranchDeepfakeNet, compute_total_loss
from deepfake_lowres.utils import ensure_dir


def move_batch_to_device(batch, device):
    return (
        batch["rgb"].to(device, non_blocking=True),
        batch["freq"].to(device, non_blocking=True),
        batch["quality_features"].to(device, non_blocking=True),
        batch["label"].to(device, non_blocking=True),
    )


def train_one_epoch(model, loader, optimizer, scaler, device, mixed_precision=True):
    model.train()
    total_loss = 0.0
    total_count = 0
    y_true, y_pred, y_score = [], [], []
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        rgb, freq, qf, labels = move_batch_to_device(batch, device)
        gate_targets = make_gate_targets_from_quality_features(qf).to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(mixed_precision and device.type == "cuda")):
            outputs = model(rgb, freq, qf)
            loss = compute_total_loss(outputs, labels, gate_targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            probs = torch.softmax(outputs["fused_logits"], dim=1)[:, 1]
            preds = outputs["fused_logits"].argmax(dim=1)
        bs = rgb.size(0)
        total_loss += loss.item() * bs
        total_count += bs
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_score.extend(probs.detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=f"{total_loss / max(total_count, 1):.4f}", acc=f"{accuracy_score(y_true, y_pred):.4f}")
    metrics = compute_multiclass_binary_metrics(y_true, y_pred, y_score)
    metrics["loss"] = float(total_loss / max(total_count, 1))
    return metrics


@torch.no_grad()
def evaluate_gated_model(model, loader, device, threshold=0.65, forced_route=None, fast_compute=False):
    model.eval()
    all_y_true, all_y_pred, all_y_score = [], [], []
    all_decisions, all_routes, all_resolutions = [], [], []
    start = time.time()
    for batch in tqdm(loader, desc="eval", leave=False):
        rgb = batch["rgb"].to(device, non_blocking=True)
        freq = batch["freq"].to(device, non_blocking=True)
        qf = batch["quality_features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        resolutions = batch["resolution"].cpu().numpy().tolist()
        if fast_compute and forced_route is not None:
            pred_out = model.predict_fast_compute(rgb, freq, qf, threshold=threshold, forced_route=forced_route)
        else:
            pred_out = model.predict(rgb, freq, qf, threshold=threshold, forced_route=forced_route)
        all_y_true.extend(labels.detach().cpu().numpy().tolist())
        all_y_pred.extend(pred_out["pred"].detach().cpu().numpy().tolist())
        all_y_score.extend(pred_out["probs"][:, 1].detach().cpu().numpy().tolist())
        all_decisions.extend(pred_out["decision"])
        all_routes.extend(pred_out["routes"].detach().cpu().numpy().tolist())
        all_resolutions.extend(resolutions)
    metrics = compute_multiclass_binary_metrics(all_y_true, all_y_pred, all_y_score)
    rc = Counter(all_routes)
    metrics.update({
        "num_samples": len(all_y_true),
        "uncertain_rate_pct": float(100.0 * sum(d == "uncertain" for d in all_decisions) / max(len(all_decisions), 1)),
        "route_usage": {
            "fast_exit1_pct": 100.0 * rc.get(0, 0) / max(len(all_routes), 1),
            "medium_exit2_pct": 100.0 * rc.get(1, 0) / max(len(all_routes), 1),
            "full_exit3_pct": 100.0 * rc.get(2, 0) / max(len(all_routes), 1),
        },
        "elapsed_sec": float(time.time() - start),
    })
    per_resolution = {}
    for res in sorted(set(all_resolutions)):
        idxs = [i for i, r in enumerate(all_resolutions) if r == res]
        y_true_res = [all_y_true[i] for i in idxs]
        y_pred_res = [all_y_pred[i] for i in idxs]
        y_score_res = [all_y_score[i] for i in idxs]
        routes_res = [all_routes[i] for i in idxs]
        decisions_res = [all_decisions[i] for i in idxs]
        rm = compute_multiclass_binary_metrics(y_true_res, y_pred_res, y_score_res)
        rr = Counter(routes_res)
        rm["uncertain_rate_pct"] = float(100.0 * sum(d == "uncertain" for d in decisions_res) / max(len(decisions_res), 1))
        rm["route_usage"] = {
            "fast_exit1_pct": 100.0 * rr.get(0, 0) / max(len(routes_res), 1),
            "medium_exit2_pct": 100.0 * rr.get(1, 0) / max(len(routes_res), 1),
            "full_exit3_pct": 100.0 * rr.get(2, 0) / max(len(routes_res), 1),
        }
        per_resolution[str(res)] = rm
    metrics["per_resolution"] = per_resolution
    return metrics


def fit_gated_model(train_df, val_df, test_df, config: dict, device):
    out_dir = ensure_dir(config.get("out_dir", "outputs/gated_dual_branch"))
    resolutions = config.get("resolutions", [128, 224, 256, 384])
    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 2))
    image_size = int(config.get("image_size_for_model", 224))
    train_ds = GatedMultiResolutionDataset(train_df, image_size=image_size, resolutions=resolutions, mode="train_random", cache_features=config.get("cache_features", False))
    val_ds = GatedMultiResolutionDataset(val_df, image_size=image_size, resolutions=resolutions, mode="eval_all", cache_features=False)
    test_ds = GatedMultiResolutionDataset(test_df, image_size=image_size, resolutions=resolutions, mode="eval_all", cache_features=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))
    model = GatedDualBranchDeepfakeNet(branch_channels=tuple(config.get("branch_channels", [12, 20, 32, 48]))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 2e-4)), weight_decay=float(config.get("weight_decay", 1e-4)))
    scaler = GradScaler(enabled=(bool(config.get("mixed_precision", True)) and device.type == "cuda"))
    ckpt_path = out_dir / config.get("ckpt_name", "best_gated_dual_branch.pt")
    best_val_auc = -1.0
    history = []
    start = time.time()
    for epoch in range(1, int(config.get("epochs", 10)) + 1):
        tm = train_one_epoch(model, train_loader, optimizer, scaler, device, config.get("mixed_precision", True))
        vm = evaluate_gated_model(model, val_loader, device, threshold=float(config.get("uncertain_threshold", 0.65)))
        row = {"Epoch": epoch, "Train Loss": tm["loss"], "Train Acc": tm["accuracy"], "Train AUC": tm["auc"], "Val Acc": vm["accuracy"], "Val AUC": vm["auc"], "Precision": vm["precision"], "Recall": vm["recall"], "F1": vm["f1"], "Uncertain Rate (%)": vm["uncertain_rate_pct"]}
        history.append(row)
        print(f"Epoch {epoch:02d} | Loss {tm['loss']:.4f} | Tr Acc {tm['accuracy']:.4f} AUC {tm['auc']:.4f} | Val Acc {vm['accuracy']:.4f} AUC {vm['auc']:.4f} | F1 {vm['f1']:.4f}")
        if vm["auc"] > best_val_auc:
            best_val_auc = vm["auc"]
            torch.save({"model": model.state_dict(), "config": config, "best_val_auc": best_val_auc}, ckpt_path)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    total_train_mins = (time.time() - start) / 60.0
    return model, pd.DataFrame(history), train_loader, val_loader, test_loader, total_train_mins
