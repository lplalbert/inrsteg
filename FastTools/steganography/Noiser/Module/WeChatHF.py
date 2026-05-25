"""WeChat 高频清零噪声层 → 注册到 Noiser"""

import torch
import torch.nn as nn

from FastTools.steganography.Noiser.Noiser import NoiseLayer, NoiseLayerManager
from FastTools.steganography.Noiser.wechat_hf_zero import WeChatHFZero


@NoiseLayerManager.register("WeChatHFZero")
class WeChatHFZeroLayer(NoiseLayer):
    """Noiser 包装：每次调用对有 watermark 的图做 WeChat 高频清零。"""

    def __init__(
        self,
        zigzag_keep: int = 21,
        canvas_h: int = 3072,
        canvas_w: int = 4096,
        crop_size: int = 256,
        random_offset: bool = True,
    ):
        super().__init__()
        self.op = WeChatHFZero(
            zigzag_keep=zigzag_keep,
            canvas_h=canvas_h,
            canvas_w=canvas_w,
            crop_size=crop_size,
            random_offset=random_offset,
        )

    def noise(self, img: torch.Tensor, cover_img: torch.Tensor):
        """img: [B, 3, H, W] watermarked → 重复铺满 canvas 做 HF 清零"""
        return self.op(img), cover_img
