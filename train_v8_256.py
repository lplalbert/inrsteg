#!/usr/bin/env python3
"""v8 fine-tuning on 256×256 / 64-bit with WeChat HF-zero noise.

从 v6_size256_msg64 的预训练权重加载，继续训练 v8 (level=4 的 FeatureGrid)。
"""

import os
import sys
from pathlib import Path

import torch
import yaml

# 确保自定义噪声层被注册
import FastTools.steganography.Noiser.Module.WeChatHF       # noqa: F401
import FastTools.steganography.Noiser.Module.ScreenShooting  # noqa: F401

from model.ismark_v8 import INRMarkTrainer


class DotDict(dict):
    def __getattr__(self, key):
        return self.get(key)
    def __setattr__(self, key, value):
        self[key] = value


def load_yaml_cfg(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def main():
    cfg_path = "./config/v8_size256_msg64.yaml"
    cfg = load_yaml_cfg(cfg_path)

    trainer = INRMarkTrainer(cfg_path)

    # 加载预训练权重 (v6_size256_msg64)
    pretrained = cfg.get("pretrained_ckpt", "")
    if pretrained and os.path.exists(pretrained):
        print(f"[train_v8_256] Loading pretrained weights from: {pretrained}")
        state = torch.load(pretrained, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]

        # Key remapping: pretrained v6 uses 'norm' for BatchNorm in LowRankFusionBlock, v8 uses 'bn'
        remapped = {}
        for k, v in state.items():
            new_k = k.replace('.norm.', '.bn.') if 'inr.' in k and '.norm.' in k else k
            remapped[new_k] = v

        # 过滤掉 shape 不匹配的 key (v8 的 struct_embedding.grids 层数不同)
        model_state = trainer.model.state_dict()
        matched = {}
        skipped = []
        for k, v in remapped.items():
            if k in model_state and v.shape == model_state[k].shape:
                matched[k] = v
            else:
                skipped.append(k)

        print(f"[train_v8_256] Matched: {len(matched)} layers, Skipped: {len(skipped)} layers")
        for s in skipped[:8]:
            print(f"  SKIP: {s}")

        model_state.update(matched)
        trainer.model.load_state_dict(model_state, strict=False)
        print("[train_v8_256] Pretrained weights loaded (strict=False).")
    else:
        print(f"[train_v8_256] WARNING: Pretrained ckpt not found: {pretrained}")
        print("[train_v8_256] Training from scratch.")

    # v8 构建 noiser 时会在构造函数中使用 progressive_noiser
    # 手动添加自定义噪声层到最后一个 stage
    trainer.model.all_noisers[-1].append(
        ("WeChatHFZero", {"zigzag_keep": 21, "canvas_h": 3072, "canvas_w": 4096, "crop_size": 256})
    )
    trainer.model.all_noisers[-1].append(
        ("ScreenShooting", {"perspective_d": 8, "moire_weight": 0.15, "light_weight": 0.85, "gauss_std": 0.0316})
    )
    print("[train_v8_256] WeChatHFZero + ScreenShooting added to final noiser stage.")

    trainer.train()


if __name__ == "__main__":
    main()
