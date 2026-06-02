from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = ConvBNReLU(in_ch, in_ch, kernel=3, stride=stride, padding=1, groups=in_ch)
        self.pw = ConvBNReLU(in_ch, out_ch, kernel=1, padding=0)

    def forward(self, x):
        return self.pw(self.dw(x))


class ConvReservoir(nn.Module):
    def __init__(self, in_ch, reservoir_ch=64, spectral_radius=0.9):
        super().__init__()
        self.input_proj = nn.Conv2d(in_ch, reservoir_ch, kernel_size=1, bias=False)
        self.input_bn = nn.BatchNorm2d(reservoir_ch)
        weights = torch.randn(reservoir_ch, reservoir_ch, 3, 3)
        weights = weights / (weights.abs().max() + 1e-8) * spectral_radius
        self.register_buffer("W_res", weights)
        self.readout = nn.Sequential(
            nn.Conv2d(reservoir_ch, reservoir_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(reservoir_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        h = F.relu(self.input_bn(self.input_proj(x)))
        h_res = torch.tanh(F.conv2d(h, self.W_res, padding=1))
        return self.readout(h + h_res)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.gate(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class MultiScaleAttentionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        b_ch = out_ch // 3
        self.branch1 = DepthwiseSeparable(in_ch, b_ch)
        self.branch2 = nn.Sequential(DepthwiseSeparable(in_ch, b_ch), DepthwiseSeparable(b_ch, b_ch))
        self.branch3 = nn.Sequential(DepthwiseSeparable(in_ch, b_ch), DepthwiseSeparable(b_ch, b_ch), DepthwiseSeparable(b_ch, b_ch))
        self.fuse = ConvBNReLU(b_ch * 3, out_ch, kernel=1, padding=0)
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        f = self.fuse(torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1))
        return self.sa(self.ca(f))


class MaDCoRN(nn.Module):
    def __init__(self, in_ch=3, num_classes=2, base_ch=32):
        super().__init__()
        c = base_ch
        self.stem = nn.Sequential(ConvBNReLU(in_ch, c, kernel=3, stride=2, padding=1), ConvBNReLU(c, c * 2, kernel=3, stride=1, padding=1))
        self.mad1 = MultiScaleAttentionBlock(c * 2, c * 4)
        self.mad2 = MultiScaleAttentionBlock(c * 4, c * 4)
        self.res1 = ConvReservoir(c * 4, c * 4)
        self.down1 = nn.Conv2d(c * 4, c * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.mad3 = MultiScaleAttentionBlock(c * 4, c * 8)
        self.mad4 = MultiScaleAttentionBlock(c * 8, c * 8)
        self.res2 = ConvReservoir(c * 8, c * 8)
        self.down2 = nn.Conv2d(c * 8, c * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.mad5 = MultiScaleAttentionBlock(c * 8, c * 16)
        self.mad6 = MultiScaleAttentionBlock(c * 16, c * 16)
        self.res3 = ConvReservoir(c * 16, c * 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c * 16, 256), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(256, num_classes))
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(self.res1(self.mad2(self.mad1(x))))
        x = self.down2(self.res2(self.mad4(self.mad3(x))))
        x = self.res3(self.mad6(self.mad5(x)))
        return self.head(self.pool(x))


def freeze_reservoir_weights(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "W_res" in name:
            param.requires_grad_(False)
