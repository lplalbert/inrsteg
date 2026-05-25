"""
V7 训练: 从 V6 checkpoint 加载权重, 用均匀坐标采样继续训练

模型架构不变, 仅数据采样改为 gen_start_coords_uniform
"""

import torch
from model.ismark_v7 import INRMarkTrainer, INRMark
from FastTools.util.TrainUtil import Args
from FastTools.light.Engine import EngineTrainer


class INRMarkTrainerV7(INRMarkTrainer):
    """V7 trainer: 从 V6 checkpoint 加载权重 (兼容不同模块路径)"""

    def build_model(self, cfg):
        model = INRMark(cfg)
        if cfg.ckpt_path:
            ckpt = torch.load(cfg.ckpt_path, map_location='cpu', weights_only=True)
            state_dict = ckpt.get('state_dict', ckpt)
            # Lightning checkpoint 的 key 可能有 "model." 前缀, 去掉
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            print(f"[V7] Loaded weights from {cfg.ckpt_path}")
        return model


INRMarkTrainerV7("./config/v7.yaml").train()
