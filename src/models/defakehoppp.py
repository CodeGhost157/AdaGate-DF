from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from deepfake_lowres.metrics import compute_binary_metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class DefakeHopPPConfig:
    image_size: int = 224
    resolutions: tuple[int, ...] = (128, 224, 256, 384)
    small_block_size: int = 13
    large_block_size: int = 32
    patch_size: int = 3
    stride: int = 2
    max_train_fit_per_class: int = 1500
    dft_num_splits: int = 31
    spatial_pca_energy: float = 0.80
    spatial_pca_max_channels: int = 10
    uncertain_low: float = 0.4
    uncertain_high: float = 0.6
    lgb_num_leaves: int = 64
    lgb_n_estimators: int = 300
    lgb_learning_rate: float = 0.05
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    random_state: int = 42


SMALL_BLOCK_CENTERS = [
    (0.32, 0.28), (0.45, 0.30), (0.68, 0.28), (0.55, 0.30),
    (0.38, 0.45), (0.62, 0.45), (0.50, 0.40), (0.50, 0.58),
]
LARGE_BLOCK_CENTERS = [(0.35, 0.35), (0.65, 0.35), (0.50, 0.68)]


def load_image_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def resize_image_np(img, size):
    return np.array(Image.fromarray(img).resize((size, size)))


def crop_square_block(img_rgb, center_xy, block_size):
    h, w = img_rgb.shape[:2]
    cx, cy = center_xy
    half = block_size // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + block_size, y1 + block_size
    out = np.zeros((block_size, block_size, 3), dtype=np.uint8)
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    dx1, dy1 = sx1 - x1, sy1 - y1
    dx2, dy2 = dx1 + (sx2 - sx1), dy1 + (sy2 - sy1)
    out[dy1:dy2, dx1:dx2] = img_rgb[sy1:sy2, sx1:sx2]
    return out


def extract_defakehoppp_blocks(img_rgb, cfg: DefakeHopPPConfig):
    h, w = img_rgb.shape[:2]
    small = [crop_square_block(img_rgb, (int(nx * w), int(ny * h)), cfg.small_block_size) for nx, ny in SMALL_BLOCK_CENTERS]
    large = [crop_square_block(img_rgb, (int(nx * w), int(ny * h)), cfg.large_block_size) for nx, ny in LARGE_BLOCK_CENTERS]
    return {"small_blocks": small, "large_blocks": large}


def extract_patches_from_block(block, patch_size=3, stride=1):
    h, w, _ = block.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(block[y:y + patch_size, x:x + patch_size, :].reshape(-1))
    return np.array(patches, dtype=np.float32)


def collect_training_patches(train_df, cfg: DefakeHopPPConfig):
    real_paths = train_df[train_df["label"] == 0]["image_path"].tolist()[: cfg.max_train_fit_per_class]
    fake_paths = train_df[train_df["label"] == 1]["image_path"].tolist()[: cfg.max_train_fit_per_class]
    patches = []
    for path in tqdm(real_paths + fake_paths, desc="Collecting training patches"):
        try:
            img = resize_image_np(load_image_rgb(path), cfg.image_size)
            blocks = extract_defakehoppp_blocks(img, cfg)
            for block in blocks["small_blocks"] + blocks["large_blocks"]:
                p = extract_patches_from_block(block, cfg.patch_size, stride=1)
                if len(p) > 0:
                    patches.append(p)
        except Exception:
            continue
    if not patches:
        raise RuntimeError("No patches were collected. Check image paths and dataset manifest.")
    return np.concatenate(patches, axis=0)


def fit_pixelhop_pca(train_df, cfg: DefakeHopPPConfig):
    patches = collect_training_patches(train_df, cfg)
    pca = PCA(n_components=27, random_state=cfg.random_state)
    pca.fit(patches)
    return pca


def pixelhop_transform_block(block, pca_model, cfg: DefakeHopPPConfig):
    h, w, _ = block.shape
    out_h = (h - cfg.patch_size) // cfg.stride + 1
    out_w = (w - cfg.patch_size) // cfg.stride + 1
    feats = np.zeros((out_h, out_w, 27), dtype=np.float32)
    oy = 0
    for y in range(0, h - cfg.patch_size + 1, cfg.stride):
        ox = 0
        for x in range(0, w - cfg.patch_size + 1, cfg.stride):
            patch = block[y:y + cfg.patch_size, x:x + cfg.patch_size, :].reshape(1, -1).astype(np.float32)
            feats[oy, ox, :] = pca_model.transform(patch)[0]
            ox += 1
        oy += 1
    return feats


def extract_all_fmaps_for_image(img_rgb, pixelhop_pca, cfg: DefakeHopPPConfig):
    blocks = extract_defakehoppp_blocks(img_rgb, cfg)
    small_fmaps = [pixelhop_transform_block(b, pixelhop_pca, cfg) for b in blocks["small_blocks"]]
    large_fmaps = [pixelhop_transform_block(b, pixelhop_pca, cfg) for b in blocks["large_blocks"]]
    return {"small_fmaps": small_fmaps, "large_fmaps": large_fmaps}


def fit_spatial_pca_for_blocktype(feature_maps_list, energy_keep=0.80, max_channels=10, random_state=42):
    c = feature_maps_list[0].shape[-1]
    variances = [np.mean([np.var(fmap[:, :, ch]) for fmap in feature_maps_list]) for ch in range(c)]
    kept = np.argsort(variances)[::-1][:max_channels]
    channel_models, kept_channels = {}, []
    for ch in kept:
        x = np.stack([fmap[:, :, ch].reshape(-1) for fmap in feature_maps_list], axis=0)
        pca = PCA(random_state=random_state).fit(x)
        cum = np.cumsum(pca.explained_variance_ratio_)
        n_keep = max(1, int(np.searchsorted(cum, energy_keep) + 1))
        channel_models[int(ch)] = {"mean": pca.mean_.copy(), "components": pca.components_[:n_keep].copy()}
        kept_channels.append(int(ch))
    return {"kept_channels": kept_channels, "channel_models": channel_models}


def apply_spatial_pca_to_block(fmap, spatial_pca_model):
    feats = []
    for ch in spatial_pca_model["kept_channels"]:
        vec = fmap[:, :, ch].reshape(1, -1)
        mean = spatial_pca_model["channel_models"][ch]["mean"]
        comps = spatial_pca_model["channel_models"][ch]["components"]
        feats.append(((vec - mean) @ comps.T)[0])
    if len(feats) == 0:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate(feats, axis=0).astype(np.float32)


def collect_feature_maps_for_training(df, image_size, pixelhop_pca, cfg: DefakeHopPPConfig, limit=None):
    rows = df if limit is None else df.iloc[:limit]
    all_small = [[] for _ in range(8)]
    all_large = [[] for _ in range(3)]
    labels = []
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Collecting feature maps"):
        try:
            img = resize_image_np(load_image_rgb(row["image_path"]), image_size)
            out = extract_all_fmaps_for_image(img, pixelhop_pca, cfg)
            for i in range(8):
                all_small[i].append(out["small_fmaps"][i])
            for i in range(3):
                all_large[i].append(out["large_fmaps"][i])
            labels.append(int(row["label"]))
        except Exception:
            continue
    return all_small, all_large, np.array(labels, dtype=np.int32)


def fit_all_spatial_pca_models(train_df, pixelhop_pca, cfg: DefakeHopPPConfig, limit=None):
    all_small, all_large, _ = collect_feature_maps_for_training(train_df, cfg.image_size, pixelhop_pca, cfg, limit=limit)
    small_models = [fit_spatial_pca_for_blocktype(all_small[i], cfg.spatial_pca_energy, cfg.spatial_pca_max_channels, cfg.random_state) for i in range(8)]
    large_models = [fit_spatial_pca_for_blocktype(all_large[i], cfg.spatial_pca_energy, cfg.spatial_pca_max_channels, cfg.random_state) for i in range(3)]
    return small_models, large_models


def build_feature_matrix(df, image_size, pixelhop_pca, small_spatial_models, large_spatial_models, cfg: DefakeHopPPConfig):
    rows, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building X at {image_size}"):
        try:
            img = resize_image_np(load_image_rgb(row["image_path"]), image_size)
            out = extract_all_fmaps_for_image(img, pixelhop_pca, cfg)
            feats = []
            for i, fmap in enumerate(out["small_fmaps"]):
                feats.append(apply_spatial_pca_to_block(fmap, small_spatial_models[i]))
            for i, fmap in enumerate(out["large_fmaps"]):
                feats.append(apply_spatial_pca_to_block(fmap, large_spatial_models[i]))
            rows.append(np.concatenate(feats, axis=0))
            labels.append(int(row["label"]))
        except Exception:
            continue
    return np.stack(rows, axis=0), np.array(labels, dtype=np.int32)


def fit_defakehoppp_pipeline(train_df, val_df, cfg: DefakeHopPPConfig | None = None):
    cfg = cfg or DefakeHopPPConfig()
    start = time.time()
    print("Fitting PixelHop PCA...")
    pixelhop_pca = fit_pixelhop_pca(train_df, cfg)
    print("Fitting Spatial PCA...")
    small_spatial_models, large_spatial_models = fit_all_spatial_pca_models(train_df, pixelhop_pca, cfg)
    print("Building train and validation feature matrices...")
    x_train, y_train = build_feature_matrix(train_df, cfg.image_size, pixelhop_pca, small_spatial_models, large_spatial_models, cfg)
    x_val, y_val = build_feature_matrix(val_df, cfg.image_size, pixelhop_pca, small_spatial_models, large_spatial_models, cfg)
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_val_s = scaler.transform(x_val)
    clf = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=cfg.lgb_num_leaves,
        n_estimators=cfg.lgb_n_estimators,
        learning_rate=cfg.lgb_learning_rate,
        subsample=cfg.lgb_subsample,
        colsample_bytree=cfg.lgb_colsample_bytree,
        random_state=cfg.random_state,
    )
    clf.fit(x_train_s, y_train, eval_set=[(x_val_s, y_val)], eval_metric="auc")
    return {
        "pixelhop_pca": pixelhop_pca,
        "small_spatial_models": small_spatial_models,
        "large_spatial_models": large_spatial_models,
        "scaler": scaler,
        "classifier": clf,
        "image_size": cfg.image_size,
        "config": cfg,
        "training_time_minutes": (time.time() - start) / 60.0,
    }


def transform_dataframe_with_model(df, model, image_size=None):
    cfg = model["config"]
    image_size = int(image_size or model["image_size"])
    x, y = build_feature_matrix(df, image_size, model["pixelhop_pca"], model["small_spatial_models"], model["large_spatial_models"], cfg)
    return model["scaler"].transform(x), y


def evaluate_model(model, df, model_name="DeFakeHop++ Approx"):
    x, y = transform_dataframe_with_model(df, model)
    probs = model["classifier"].predict_proba(x)[:, 1]
    metrics = compute_binary_metrics(y, probs, uncertain_low=model["config"].uncertain_low, uncertain_high=model["config"].uncertain_high)
    return pd.DataFrame([{
        "Model": model_name,
        "Accuracy": metrics["accuracy"],
        "AUC": metrics["auc"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
        "Uncertain Rate (%)": metrics["uncertain_rate"],
    }])


def evaluate_resolutions(model, test_df, resolutions=(128, 224, 256, 384)):
    rows = []
    for res in resolutions:
        x, y = transform_dataframe_with_model(test_df, model, image_size=int(res))
        probs = model["classifier"].predict_proba(x)[:, 1]
        m = compute_binary_metrics(y, probs, uncertain_low=model["config"].uncertain_low, uncertain_high=model["config"].uncertain_high)
        rows.append({"Resolution": int(res), "Accuracy": m["accuracy"], "AUC": m["auc"], "Uncertain Rate (%)": m["uncertain_rate"]})
    return pd.DataFrame(rows)
