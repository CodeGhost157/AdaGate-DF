from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .transforms import build_rgb_transforms, build_single_resolution_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BinaryImageDataset(Dataset):
    def __init__(self, dataframe, image_size: int = 224, train: bool = True):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = build_single_resolution_transforms(image_size=image_size, train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        label = float(row["label"])
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise FileNotFoundError(f"Could not read image: {path}") from exc
        return {
            "image": self.transform(image),
            "label": torch.tensor(label, dtype=torch.float32),
            "path": path,
        }


class MultiResolutionImageDataset(Dataset):
    def __init__(self, dataframe, resolutions=(128, 224, 256, 384), image_size_for_model: int = 224, mode: str = "train_random"):
        self.df = dataframe.reset_index(drop=True).copy()
        self.resolutions = list(resolutions)
        self.image_size_for_model = image_size_for_model
        self.mode = mode
        self.train = mode == "train_random"
        self.transforms = {
            res: build_rgb_transforms(res, image_size_for_model, train=self.train, strong=False)
            for res in self.resolutions
        }
        self.strong_real_transforms = {
            res: build_rgb_transforms(res, image_size_for_model, train=True, strong=True)
            for res in self.resolutions
        }
        if mode == "eval_all":
            self.samples = [(idx, res) for idx in range(len(self.df)) for res in self.resolutions]
        else:
            self.samples = [(idx, None) for idx in range(len(self.df))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row_idx, res = self.samples[idx]
        row = self.df.iloc[row_idx]
        path = row["image_path"]
        label = float(row["label"])
        is_oversampled_real = bool(row.get("is_oversampled_real", False))
        if res is None:
            res = random.choice(self.resolutions)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise FileNotFoundError(f"Could not read image: {path}") from exc
        transform = self.strong_real_transforms[res] if self.train and is_oversampled_real else self.transforms[res]
        return {
            "image": transform(image),
            "label": torch.tensor(label, dtype=torch.float32),
            "path": path,
            "resolution": torch.tensor(res, dtype=torch.long),
        }


class FixedResolutionImageDataset(Dataset):
    def __init__(self, dataframe, resolution: int, image_size_for_model: int = 224):
        self.df = dataframe.reset_index(drop=True).copy()
        self.resolution = int(resolution)
        self.transform = build_rgb_transforms(resolution, image_size_for_model, train=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        label = float(row["label"])
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise FileNotFoundError(f"Could not read image: {path}") from exc
        return {
            "image": self.transform(image),
            "label": torch.tensor(label, dtype=torch.float32),
            "path": path,
            "resolution": torch.tensor(self.resolution, dtype=torch.long),
        }
