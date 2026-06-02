from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def compute_blur_and_sharpness_from_rgb(np_rgb):
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(lap))
    blur_proxy = 1.0 / (sharpness + 1e-6)
    return blur_proxy, sharpness


def compute_jpeg_compression_proxy(path):
    file_size_kb = os.path.getsize(path) / 1024.0
    ext = Path(path).suffix.lower()
    return float(1.0 / (file_size_kb + 1e-6)) if ext in [".jpg", ".jpeg"] else 0.0


def compute_dct_tensor(pil_img, out_size=224):
    gray = np.array(pil_img.convert("L")).astype(np.float32) / 255.0
    dct = cv2.dct(gray)
    dct = np.log1p(np.abs(dct))
    dct = (dct - dct.min()) / (dct.max() - dct.min() + 1e-6)
    dct = cv2.resize(dct, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(dct, dtype=torch.float32).unsqueeze(0)


def build_quality_features(pil_img, path, resized_resolution):
    np_rgb = np.array(pil_img.convert("RGB"))
    blur_proxy, sharpness = compute_blur_and_sharpness_from_rgb(np_rgb)
    compression_proxy = compute_jpeg_compression_proxy(path)
    feat = np.array([
        resized_resolution / 224.0,
        min(blur_proxy * 100.0, 10.0),
        min(sharpness / 500.0, 10.0),
        min(compression_proxy * 50.0, 10.0),
        np.mean(np_rgb) / 255.0,
        np.std(np_rgb) / 255.0,
    ], dtype=np.float32)
    return feat


def make_gate_targets_from_quality_features(qf_batch):
    targets = []
    qf_np = qf_batch.detach().cpu().numpy()
    for qf in qf_np:
        res_norm, blur_norm, sharp_norm, compression_norm, brightness, contrast = qf
        approx_res = res_norm * 224.0
        if approx_res >= 160 and sharp_norm >= 0.18 and compression_norm <= 0.8:
            targets.append(0)
        elif approx_res >= 128 and sharp_norm >= 0.06:
            targets.append(1)
        else:
            targets.append(2)
    return torch.tensor(targets, dtype=torch.long)


class GatedMultiResolutionDataset(Dataset):
    def __init__(self, dataframe, image_size=224, resolutions=(128, 224, 256, 384), mode="train_random", cache_features=False):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_size = int(image_size)
        self.resolutions = list(resolutions)
        self.mode = mode
        self.cache_features = cache_features
        self.cache = {}
        if mode == "eval_all":
            self.samples = [(idx, res) for idx in range(len(self.df)) for res in self.resolutions]
        else:
            self.samples = [(idx, None) for idx in range(len(self.df))]
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def pil_to_rgb_tensor(self, pil_img):
        arr = np.array(pil_img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).float()
        return (tensor - self.mean) / self.std

    def _augment_oversampled_real(self, img):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.35:
            img = img.rotate(random.uniform(-12, 12))
        if random.random() < 0.35:
            factor = random.uniform(0.85, 1.15)
            arr = np.array(img).astype(np.float32)
            img = Image.fromarray(np.clip(arr * factor, 0, 255).astype(np.uint8))
        return img

    def __getitem__(self, idx):
        row_idx, res = self.samples[idx]
        row = self.df.iloc[row_idx]
        path = row["image_path"]
        label = int(row["label"])
        is_oversampled_real = bool(row.get("is_oversampled_real", False))
        if res is None:
            res = random.choice(self.resolutions)
        cache_key = (path, res, is_oversampled_real)
        if self.cache_features and cache_key in self.cache:
            rgb, freq, quality_feat = self.cache[cache_key]
        else:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image_res = image.resize((res, res), Image.BILINEAR)
            if self.mode == "train_random" and is_oversampled_real:
                image_res = self._augment_oversampled_real(image_res)
            quality_feat_np = build_quality_features(image_res, path, res)
            image_model = image_res.resize((self.image_size, self.image_size), Image.BILINEAR)
            rgb = self.pil_to_rgb_tensor(image_model)
            freq = compute_dct_tensor(image_model, out_size=self.image_size)
            quality_feat = torch.tensor(quality_feat_np, dtype=torch.float32)
            if self.cache_features:
                self.cache[cache_key] = (rgb, freq, quality_feat)
        return {
            "rgb": rgb,
            "freq": freq,
            "quality_features": quality_feat,
            "label": torch.tensor(label, dtype=torch.long),
            "resolution": torch.tensor(res, dtype=torch.long),
            "path": path,
        }
