from __future__ import annotations

import torch.nn as nn
import timm


class BinaryTimmDeepfakeDetector(nn.Module):
    def __init__(self, backbone_name: str = "tf_efficientnet_b4.ns_jft_in1k", dropout: float = 0.2, pretrained: bool = True):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.encoder.num_features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, 1))

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features).squeeze(1)
