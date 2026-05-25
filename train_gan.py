"""GAN 训练：v6 + PatchGAN，无噪声层"""

import os
import torch

# 导入以注册噪声模块
import FastTools.steganography.Noiser.Module.WeChatHF       # noqa
import FastTools.steganography.Noiser.Module.ScreenShooting  # noqa

from model.ismark_v6_30bit import INRMarkTrainer
from model.ismark_v6_gan import INRMark as GANModel


class GANTrainer(INRMarkTrainer):
    """继承 v6 的 Trainer（含 build_dataset），替换 build_model 用 GAN 模型"""

    def build_model(self, cfg):
        model = GANModel(cfg)
        if cfg.ckpt_path and os.path.exists(cfg.ckpt_path):
            state = torch.load(cfg.ckpt_path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            model_state = model.state_dict()
            matched = 0
            for k, v in state.items():
                if k in model_state and v.shape == model_state[k].shape:
                    model_state[k] = v
                    matched += 1
            model.load_state_dict(model_state, strict=False)
            print(f"[GAN] Loaded {matched} params from {cfg.ckpt_path}")
        # 阻止引擎和 Lightning 二次加载（新模块参数不在旧 ckpt 中）
        cfg.ckpt_path = None
        self.ckpt_path = None
        return model


def main():
    GANTrainer("./config/v6_gan.yaml").train()


if __name__ == "__main__":
    main()
