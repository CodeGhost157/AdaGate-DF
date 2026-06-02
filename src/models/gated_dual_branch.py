from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, kernel_size=3, stride=stride, groups=in_ch)
        self.pw = ConvBNAct(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.pw(self.dw(x))


class ExitHead(nn.Module):
    def __init__(self, in_ch, hidden_dim=64, num_classes=2, dropout=0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(in_ch, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        feat = self.pool(x).flatten(1)
        logits = self.fc(feat)
        return logits, feat


class AdaptiveGate(nn.Module):
    def __init__(self, in_features=6, hidden_dim=32, num_routes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_routes),
        )

    def forward(self, qf):
        logits = self.net(qf)
        probs = torch.softmax(logits, dim=1)
        route_idx = probs.argmax(dim=1)
        return logits, probs, route_idx


class EarlyExitBranch(nn.Module):
    def __init__(self, in_ch, channels=(12, 20, 32, 48), num_classes=2, exit_hidden=48):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.stem = ConvBNAct(in_ch, c1, kernel_size=3, stride=2)
        self.stage1 = nn.Sequential(DepthwiseSeparableBlock(c1, c2, stride=2), DepthwiseSeparableBlock(c2, c2, stride=1))
        self.stage2 = nn.Sequential(DepthwiseSeparableBlock(c2, c3, stride=2), DepthwiseSeparableBlock(c3, c3, stride=1))
        self.stage3 = nn.Sequential(DepthwiseSeparableBlock(c3, c4, stride=2), DepthwiseSeparableBlock(c4, c4, stride=1))
        self.exit1 = ExitHead(c2, hidden_dim=exit_hidden, num_classes=num_classes)
        self.exit2 = ExitHead(c3, hidden_dim=exit_hidden, num_classes=num_classes)
        self.exit3 = ExitHead(c4, hidden_dim=exit_hidden, num_classes=num_classes)

    def forward_all(self, x):
        x = self.stem(x)
        x1 = self.stage1(x)
        e1_logits, e1_feat = self.exit1(x1)
        x2 = self.stage2(x1)
        e2_logits, e2_feat = self.exit2(x2)
        x3 = self.stage3(x2)
        e3_logits, e3_feat = self.exit3(x3)
        return {"exit1_logits": e1_logits, "exit2_logits": e2_logits, "exit3_logits": e3_logits, "deep_feat": e3_feat}

    def forward_route(self, x, route):
        x = self.stem(x)
        x1 = self.stage1(x)
        if route == 0:
            return self.exit1(x1)
        x2 = self.stage2(x1)
        if route == 1:
            return self.exit2(x2)
        x3 = self.stage3(x2)
        return self.exit3(x3)


class GatedDualBranchDeepfakeNet(nn.Module):
    def __init__(self, branch_channels=(12, 20, 32, 48), gate_features=6, num_classes=2):
        super().__init__()
        self.spatial = EarlyExitBranch(3, channels=branch_channels, num_classes=num_classes, exit_hidden=48)
        self.frequency = EarlyExitBranch(1, channels=branch_channels, num_classes=num_classes, exit_hidden=48)
        self.gate = AdaptiveGate(in_features=gate_features, hidden_dim=32, num_routes=3)
        deep_dim = branch_channels[-1]
        self.fusion = nn.Sequential(nn.Linear(deep_dim * 2, 96), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(96, num_classes))

    def forward(self, rgb, freq, qf):
        gate_logits, gate_probs, gate_routes = self.gate(qf)
        spatial_out = self.spatial.forward_all(rgb)
        freq_out = self.frequency.forward_all(freq)
        fused_feat = torch.cat([spatial_out["deep_feat"], freq_out["deep_feat"]], dim=1)
        fused_logits = self.fusion(fused_feat)
        return {
            "gate_logits": gate_logits,
            "gate_probs": gate_probs,
            "route_idx": gate_routes,
            "spatial_exit1": spatial_out["exit1_logits"],
            "spatial_exit2": spatial_out["exit2_logits"],
            "spatial_exit3": spatial_out["exit3_logits"],
            "freq_exit1": freq_out["exit1_logits"],
            "freq_exit2": freq_out["exit2_logits"],
            "freq_exit3": freq_out["exit3_logits"],
            "fused_logits": fused_logits,
        }

    def predict_fast_compute(self, rgb, freq, qf, threshold=0.65, forced_route=None):
        if forced_route is None:
            _, _, routes = self.gate(qf)
            if len(torch.unique(routes)) > 1:
                return self.predict(rgb, freq, qf, threshold=threshold, forced_route=None)
            route = int(routes[0].item())
        else:
            route = int(forced_route)
            routes = torch.full((rgb.size(0),), route, dtype=torch.long, device=rgb.device)
        if route in [0, 1]:
            s_logits, _ = self.spatial.forward_route(rgb, route)
            f_logits, _ = self.frequency.forward_route(freq, route)
            final_logits = 0.5 * s_logits + 0.5 * f_logits
        else:
            _, s_feat = self.spatial.forward_route(rgb, 2)
            _, f_feat = self.frequency.forward_route(freq, 2)
            final_logits = self.fusion(torch.cat([s_feat, f_feat], dim=1))
        return self._decision(final_logits, routes, threshold)

    def predict(self, rgb, freq, qf, threshold=0.65, forced_route=None):
        out = self.forward(rgb, freq, qf)
        batch_size = rgb.size(0)
        routes = out["route_idx"] if forced_route is None else torch.full((batch_size,), forced_route, dtype=torch.long, device=rgb.device)
        final_logits = []
        for i in range(batch_size):
            route = int(routes[i].item())
            if route == 0:
                logits_i = 0.5 * out["spatial_exit1"][i:i+1] + 0.5 * out["freq_exit1"][i:i+1]
            elif route == 1:
                logits_i = 0.5 * out["spatial_exit2"][i:i+1] + 0.5 * out["freq_exit2"][i:i+1]
            else:
                logits_i = out["fused_logits"][i:i+1]
            final_logits.append(logits_i)
        final_logits = torch.cat(final_logits, dim=0)
        decision = self._decision(final_logits, routes, threshold)
        decision["raw_outputs"] = out
        return decision

    @staticmethod
    def _decision(final_logits, routes, threshold):
        probs = torch.softmax(final_logits, dim=1)
        conf, pred = probs.max(dim=1)
        decision = []
        for c, p in zip(conf.tolist(), pred.tolist()):
            if c < threshold:
                decision.append("uncertain")
            elif p == 0:
                decision.append("real")
            else:
                decision.append("deepfake")
        return {"logits": final_logits, "probs": probs, "pred": pred, "conf": conf, "decision": decision, "routes": routes}


def compute_total_loss(outputs, labels, gate_targets, gate_weight=0.20):
    loss = 0.0
    loss += 0.08 * F.cross_entropy(outputs["spatial_exit1"], labels)
    loss += 0.12 * F.cross_entropy(outputs["spatial_exit2"], labels)
    loss += 0.15 * F.cross_entropy(outputs["spatial_exit3"], labels)
    loss += 0.08 * F.cross_entropy(outputs["freq_exit1"], labels)
    loss += 0.12 * F.cross_entropy(outputs["freq_exit2"], labels)
    loss += 0.15 * F.cross_entropy(outputs["freq_exit3"], labels)
    loss += 0.30 * F.cross_entropy(outputs["fused_logits"], labels)
    loss += gate_weight * F.cross_entropy(outputs["gate_logits"], gate_targets)
    return loss
