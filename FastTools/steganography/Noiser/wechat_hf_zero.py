"""
WeChat 高频清零噪声层（可微 torch 实现）。

模拟微信发送图片时的 JPEG 预处理：
  RGB → YCbCr(BT.601) → 4:2:0 → 8×8 DCT → 之字形高频清零 → IDCT → RGB

流程:
  1. 将 batch 图片拼成大 canvas (默认 4096×3072)，每张加入随机偏移
  2. 对大图做 DCT 域高频清零
  3. 从大图裁回原始尺寸

所有操作均使用 torch 算子，完全可微。
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ 8×8 DCT / IDCT 矩阵 ============

def build_dct_matrix(n: int = 8) -> torch.Tensor:
    """构造 n×n DCT-II 正交矩阵 (Jpeg 标准)。"""
    i = torch.arange(n, dtype=torch.float32)
    j = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    D = torch.cos(math.pi / n * (i + 0.5) * j)
    D[0] *= 1.0 / math.sqrt(2.0)
    D *= math.sqrt(2.0 / n)
    return D  # [n, n]


# ============ 之字形扫描序 ============

JPEG_ZZ = (
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
)


def make_zigzag_mask(zigzag_keep: int) -> torch.Tensor:
    """生成 8×8 二进制掩码，前 zigzag_keep 个之字形系数为 1，其余为 0。"""
    mask_1d = torch.zeros(64)
    mask_1d[:zigzag_keep] = 1.0
    mask_2d = torch.zeros(8, 8)
    for k, idx in enumerate(JPEG_ZZ):
        mask_2d[idx // 8, idx % 8] = mask_1d[k]
    return mask_2d


# ============ 色彩空间 ============

def rgb_to_ycbcr(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BT.601: [..., 3] 或 [..., H, W]"""
    r, g, b = rgb[..., 0:1, :, :], rgb[..., 1:2, :, :], rgb[..., 2:3, :, :]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + (128.0 / 255.0)
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + (128.0 / 255.0)
    return y, cb, cr


def ycbcr_to_rgb(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    r = y + 1.402 * (cr - 128.0 / 255.0)
    g = y - 0.344136 * (cb - 128.0 / 255.0) - 0.714136 * (cr - 128.0 / 255.0)
    b = y + 1.772 * (cb - 128.0 / 255.0)
    return torch.cat([r, g, b], dim=-3)


def chroma_420_down(cb: torch.Tensor, cr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """2×2 均值池化 → 4:2:0"""
    cb2 = F.avg_pool2d(cb, kernel_size=2, stride=2, padding=0)
    cr2 = F.avg_pool2d(cr, kernel_size=2, stride=2, padding=0)
    return cb2, cr2


def chroma_420_up(cb2: torch.Tensor, cr2: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """最近邻上采样回原始尺寸"""
    cb_up = F.interpolate(cb2, size=(h, w), mode='nearest')
    cr_up = F.interpolate(cr2, size=(h, w), mode='nearest')
    return cb_up, cr_up


# ============ 分块 DCT / IDCT ============

def block_dct_2d(x: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """对 [N, C, H, W] 的每个 8×8 块做 DCT。
    返回同 shape tensor。
    """
    N, C, H, W = x.shape
    assert H % 8 == 0 and W % 8 == 0, f"DCT 要求 H,W 为 8 的倍数, got {H}×{W}"

    # unfold 提取 8×8 patches: [N, C*64, n_blocks_h, n_blocks_w]
    patches = F.unfold(x, kernel_size=8, stride=8)  # [N, C*64, L]
    patches = patches.view(N, C, 64, -1)  # [N, C, 64, L]
    patches = patches.permute(0, 1, 3, 2)  # [N, C, L, 64]
    patches = patches.reshape(N * C, -1, 8, 8)  # [N*C*L, 8, 8]

    # DCT: D @ block @ D^T
    dct = D.to(x.device) @ patches @ D.to(x.device).T  # [N*C*L, 8, 8]

    # Reshape back
    n_blocks = H // 8 * W // 8
    dct = dct.reshape(N, C, n_blocks, 8, 8)
    return dct  # [N, C, n_blocks, 8, 8]


def block_idct_2d(dct: torch.Tensor, D: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """block_dct_2d 的逆操作。dct: [N, C, n_blocks, 8, 8] → [N, C, H, W]"""
    N, C, n_blocks, _, _ = dct.shape
    device = dct.device
    D_dev = D.to(device)

    patches = dct.reshape(N * C, n_blocks, 8, 8)  # [N*C, L, 8, 8]
    idct = D_dev.T @ patches @ D_dev  # [N*C, L, 8, 8]

    # Fold back: patches → image
    idct = idct.reshape(N * C, n_blocks, 64)  # [N*C, L, 64]
    idct = idct.permute(0, 2, 1)  # [N*C, 64, L]
    idct = idct.reshape(N, C * 64, n_blocks)
    img = F.fold(idct, output_size=(H, W), kernel_size=8, stride=8)
    return img  # [N, C, H, W]


# ============ 主模块 ============

class WeChatHFZero(nn.Module):
    """微信 JPEG 高频清零噪声层（可微）。

    Args:
        zigzag_keep: DCT 之字形保留系数数 (1~64)
        canvas_h: 拼图高度
        canvas_w: 拼图宽度
        crop_size: 单个 crop 尺寸
        random_offset: 拼图时是否给每张图加随机偏移
    """

    def __init__(
        self,
        zigzag_keep: int = 21,
        canvas_h: int = 3072,
        canvas_w: int = 4096,
        crop_size: int = 256,
        random_offset: bool = True,
    ):
        super().__init__()
        self.zigzag_keep = zigzag_keep
        self.canvas_h = canvas_h
        self.canvas_w = canvas_w
        self.crop_size = crop_size
        self.random_offset = random_offset

        # 预存 DCT 矩阵和掩码
        self.register_buffer('D', build_dct_matrix(8))
        mask = make_zigzag_mask(zigzag_keep)
        self.register_buffer('hf_mask', mask)

    def _hf_zero_plane(self, plane: torch.Tensor) -> torch.Tensor:
        """对单通道 plane [N, 1, H, W] 做分块 DCT + 高频清零 + IDCT。"""
        N, C, H, W = plane.shape
        assert C == 1
        dct_blocks = block_dct_2d(plane, self.D)  # [N, 1, L, 8, 8]
        dct_blocks = dct_blocks * self.hf_mask.view(1, 1, 1, 8, 8)
        return block_idct_2d(dct_blocks, self.D, H, W)

    def forward(self, x: torch.Tensor, cover: torch.Tensor = None) -> torch.Tensor:
        """x: [B, 3, crop_size, crop_size] → [B, 3, crop_size, crop_size]"""
        B, C, H, W = x.shape
        device = x.device

        # ---- 1. 重复水印图铺满整个 canvas（每像素都有水印） ----
        tiles_per_row = self.canvas_w // self.crop_size
        tiles_per_col = self.canvas_h // self.crop_size
        tiles_total = tiles_per_row * tiles_per_col  # 16×12 = 192

        repeat_times = (tiles_total + B - 1) // B          # 192/8 = 24
        x_pad = x.repeat(repeat_times, 1, 1, 1)[:tiles_total]  # [192, 3, 256, 256]

        # reshape 直接平铺成 canvas: [192,3,256,256] → [1, 3, 3072, 4096]
        canvas = (
            x_pad
            .reshape(tiles_per_col, tiles_per_row, C, self.crop_size, self.crop_size)
            .permute(2, 0, 3, 1, 4)    # [C, 12, 256, 16, 256]
            .reshape(1, C, self.canvas_h, self.canvas_w)
        )

        # ---- 2. RGB → YCbCr → 4:2:0 ----
        y, cb, cr = rgb_to_ycbcr(canvas)           # all [1, 1, H, W]
        cb2, cr2 = chroma_420_down(cb, cr)          # [1, 1, H/2, W/2]

        # ---- 3. 高频清零（Y + Cb + Cr 各平面）----
        y_q = self._hf_zero_plane(y)
        cb_q = self._hf_zero_plane(cb2)
        cr_q = self._hf_zero_plane(cr2)

        # ---- 4. 色度上采样 → YCbCr → RGB ----
        cb_up, cr_up = chroma_420_up(cb_q, cr_q, self.canvas_h, self.canvas_w)
        canvas_out = ycbcr_to_rgb(y_q, cb_up, cr_up)  # [1, 3, H, W]
        canvas_out = canvas_out.clamp(0, 1)

        # ---- 5. 裁回 8 张水印图（取 canvas 最左上的 8 个 tile） ----
        # canvas_out: [1, 3, 3072, 4096]
        # reshape → [12, 16, 3, 256, 256] → 取前 flattend 8 个
        tiles_out = (
            canvas_out
            .reshape(1, C, tiles_per_col, self.crop_size, tiles_per_row, self.crop_size)
            .permute(0, 2, 4, 1, 3, 5)                              # [1, 12, 16, 3, 256, 256]
            .reshape(tiles_total, C, self.crop_size, self.crop_size)  # [192, 3, 256, 256]
        )

        return tiles_out[:B]  # [B, 3, 256, 256]


# ============ 测试 ============

if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = WeChatHFZero(zigzag_keep=21, canvas_h=3072, canvas_w=4096, crop_size=256)
    layer = layer.to(device)

    x = torch.rand(8, 3, 256, 256, device=device)
    t0 = time.time()
    with torch.no_grad():
        y = layer(x)
    t1 = time.time()
    print(f"Input:  {x.shape}  range [{x.min():.3f}, {x.max():.3f}]")
    print(f"Output: {y.shape}  range [{y.min():.3f}, {y.max():.3f}]")
    print(f"Time:   {(t1 - t0) * 1000:.1f} ms")
    print(f"Max diff: {(y - x).abs().max():.6f}")

    # 测试梯度
    x_grad = torch.randn(8, 3, 256, 256, device=device, requires_grad=True)
    layer.random_offset = False  # 关随机以便梯度稳定
    y_grad = layer(x_grad)
    loss = y_grad.sum()
    loss.backward()
    print(f"Gradient OK: grad shape={x_grad.grad.shape}, max={x_grad.grad.abs().max():.4f}")
