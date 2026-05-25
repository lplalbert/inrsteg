"""ScreenShooting 噪声层 — 模拟相机拍摄屏幕的畸变 (来自 PIMOG)。"""

import random

import numpy as np
import torch
import torch.nn as nn
import kornia

from FastTools.steganography.Noiser.Noiser import NoiseLayer, NoiseLayerManager


# ============ affine helpers ============

def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    matrix = torch.eye(3, device=translation.device, dtype=translation.dtype)
    matrix = matrix.repeat(translation.shape[0], 1, 1)
    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix


def _compute_rotation_matrix(angle: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    scale = torch.ones((angle.shape[0], 2), device=angle.device, dtype=angle.dtype)
    return kornia.get_rotation_matrix2d(center, angle, scale)


# ============ distortion ops ============

def perspective(image: torch.Tensor, d: int = 8) -> torch.Tensor:
    """随机透视变换, 四个角点随机偏移 ±d 像素。"""
    B, C, H, W = image.shape
    device = image.device
    points_src = torch.tensor([
        [0., 0.], [W - 1., 0.], [W - 1., H - 1.], [0., H - 1.],
    ], device=device).unsqueeze(0).expand(B, -1, -1)

    points_dst = points_src.clone()
    offset = (torch.rand(B, 4, 2, device=device) * 2 - 1) * d
    points_dst = points_dst + offset

    M = kornia.geometry.get_perspective_transform(points_src, points_dst)
    return kornia.geometry.warp_perspective(image.float(), M, dsize=(H, W))


def light_distortion(images: torch.Tensor, mode: int = 1) -> torch.Tensor:
    """光照不均匀畸变: 线性渐变 或 径向渐变。"""
    B, C, H, W = images.shape
    device = images.device

    a = 0.7 + torch.rand(1, device=device) * 0.2
    b = 1.1 + torch.rand(1, device=device) * 0.2

    if mode == 0:
        # 线性渐变，随机方向
        direction = random.randint(0, 3)
        row = torch.linspace(0, 1, H, device=device)
        mask_2d = (b - a) * row + a  # [H]
        mask_2d = mask_2d.unsqueeze(1).expand(H, W)
        mask_2d = torch.rot90(mask_2d, direction)
    else:
        # 径向渐变，随机光源位置
        x = random.randint(0, W - 1)
        y = random.randint(0, H - 1)
        max_len = max(
            np.sqrt(x**2 + y**2),
            np.sqrt((x - (W-1))**2 + y**2),
            np.sqrt(x**2 + (y - (H-1))**2),
            np.sqrt((x - (W-1))**2 + (y - (H-1))**2),
        ) + 1e-8
        i = torch.arange(H, device=device).float()
        j = torch.arange(W, device=device).float()
        ii, jj = torch.meshgrid(i, j, indexing='ij')
        distances = torch.sqrt((ii - y)**2 + (jj - x)**2)
        mask_2d = distances / max_len * (a - b) + b

    return mask_2d.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)


def moire_distortion(images: torch.Tensor) -> torch.Tensor:
    """摩尔纹畸变，3 通道各生成随机干涉条纹。"""
    B, C, H, W = images.shape
    device = images.device
    moire = torch.zeros_like(images)

    i = torch.arange(H, device=device).float()
    j = torch.arange(W, device=device).float()
    ii, jj = torch.meshgrid(i, j, indexing='ij')

    for ch in range(C):
        theta = torch.tensor(random.uniform(0, np.pi), device=device)
        cx = torch.tensor(random.random() * W, device=device)
        cy = torch.tensor(random.random() * H, device=device)
        dist = torch.sqrt((ii - cy)**2 + (jj - cx)**2)
        z1 = 0.5 + 0.5 * torch.cos(2 * np.pi * dist / max(H, W) * 4)
        z2 = 0.5 + 0.5 * torch.cos(
            torch.cos(theta) * jj * 0.12 + torch.sin(theta) * ii * 0.12
        )
        moire[:, ch, :, :] = torch.minimum(z1, z2)

    return moire * 2 - 1  # 映射到 [-1, 1]


# ============ NoiseLayer 注册 ============

@NoiseLayerManager.register("ScreenShooting")
class ScreenShootingLayer(NoiseLayer):
    """相机拍摄屏幕噪声层: 透视畸变 + 光照变化 + 摩尔纹 + 高斯噪声。

    Args:
        perspective_d: 透视偏移像素范围
        light_mode: 0=线性渐变, 1=径向渐变, -1=随机
        moire_weight: 摩尔纹混合权重
        light_weight: 光照混合权重
        gauss_std: 高斯噪声标准差 (默认 sqrt(0.001) ≈ 0.032)
    """

    def __init__(
        self,
        perspective_d: int = 8,
        light_mode: int = -1,
        moire_weight: float = 0.15,
        light_weight: float = 0.85,
        gauss_std: float = 0.0316,
    ):
        super().__init__()
        self.perspective_d = perspective_d
        self.light_mode = light_mode
        self.moire_weight = moire_weight
        self.light_weight = light_weight
        self.gauss_std = gauss_std

    def noise(self, img: torch.Tensor, cover_img: torch.Tensor):
        """img: [B, 3, H, W] watermarked → (noised, cover_img)"""
        device = img.device
        # 透视畸变
        noised = perspective(img, self.perspective_d)

        # 光照畸变
        lm = self.light_mode
        if lm < 0:
            lm = random.randint(0, 1)
        L = light_distortion(noised, lm)

        # 摩尔纹
        Z = moire_distortion(noised)

        # 混合
        noised = noised * L * self.light_weight + Z * self.moire_weight

        # 高斯噪声
        noised = noised + self.gauss_std * torch.randn_like(noised)

        return noised.clamp(0, 1), cover_img
