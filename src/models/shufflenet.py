from __future__ import annotations

import torch.nn as nn
from torchvision import models


class ShuffleNetDeepfakeDetector(nn.Module):
    def __init__(self, dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        self.backbone = models.shufflenet_v2_x1_0(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    def forward(self, x):
        return self.backbone(x).squeeze(1)
