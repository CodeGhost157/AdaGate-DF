from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_rgb_transforms(
    source_resolution: int = 224,
    image_size_for_model: int = 224,
    train: bool = True,
    strong: bool = False,
):
    if train and strong:
        return transforms.Compose([
            transforms.Resize((source_resolution, source_resolution)),
            transforms.Resize((image_size_for_model + 32, image_size_for_model + 32)),
            transforms.RandomResizedCrop(image_size_for_model, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.04),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    if train:
        return transforms.Compose([
            transforms.Resize((source_resolution, source_resolution)),
            transforms.Resize((image_size_for_model + 16, image_size_for_model + 16)),
            transforms.RandomCrop(image_size_for_model),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((source_resolution, source_resolution)),
        transforms.Resize((image_size_for_model, image_size_for_model)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_single_resolution_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random"),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
