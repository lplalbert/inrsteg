"""
v6 + PatchGAN：对抗训练替代 MSE 图像损失。

不加噪声层，只用 msg_loss + GAN adversarial loss 互制衡。

改编自 videoseal (Meta) 的 NLayerDiscriminator + hinge loss + adopt_weight。
"""

import copy
import functools
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接 copy v6 的 INRMark（保留所有架构和训练逻辑）
from model.ismark_v6_30bit import INRMark as INRMarkV6, get_cfg_value


# ============ PatchGAN Discriminator (from videoseal) ============

class NLayerDiscriminator(nn.Module):
    """PatchGAN，输入 [B,3,H,W]，输出 [B,1,h,w] 分数图"""

    def __init__(self, input_nc=3, ndf=32, n_layers=3, use_actnorm=False):
        super().__init__()
        if not use_actnorm:
            norm_layer = lambda ndf: nn.GroupNorm(4, ndf)
        else:
            norm_layer = lambda ndf: nn.BatchNorm2d(ndf)
        use_bias = norm_layer != nn.GroupNorm

        kw, padw = 4, 1
        seq = [nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            seq += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        seq += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=padw),
        ]
        self.main = nn.Sequential(*seq)

    def forward(self, x):
        return self.main(x)


# ============ losses ============

def hinge_d_loss(logits_real, logits_fake):
    loss_real = F.relu(1. - logits_real).mean()
    loss_fake = F.relu(1. + logits_fake).mean()
    return 0.5 * (loss_real + loss_fake)


def adopt_weight(weight, global_step, threshold=0, value=0.):
    return value if global_step < threshold else weight


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ============ GAN 版本的 INRMark ============

class INRMark(INRMarkV6):
    """v6 + PatchGAN。不加噪声层，msg_loss + GAN adversarial loss 互制衡。"""

    def __init__(self, args):
        super().__init__(args)

        # GAN 配置（用 args.get 替代 getattr，避免 DotDict 返回 None 掩盖默认值）
        self.w_img = args.get('w_img', 0.0) if hasattr(args, 'get') else getattr(args, 'w_img', 0.0)
        self.w_gan = args.get('w_gan', 0.01) if hasattr(args, 'get') else getattr(args, 'w_gan', 0.01)
        self.gan_start = args.get('gan_start', 0) if hasattr(args, 'get') else getattr(args, 'gan_start', 0)
        self.gan_warmup = args.get('gan_warmup', 0) if hasattr(args, 'get') else getattr(args, 'gan_warmup', 0)

        # 判别器
        self.discriminator = NLayerDiscriminator(
            input_nc=3,
            n_layers=args.get('gan_n_layers', 4) if hasattr(args, 'get') else getattr(args, 'gan_n_layers', 4),
            ndf=args.get('gan_ndf', 64) if hasattr(args, 'get') else getattr(args, 'gan_ndf', 64),
        ).apply(weights_init)

    def build_optimizers(self, args):
        lr = float(args.get('lr', 4e-4) if hasattr(args, 'get') else getattr(args, 'lr', 4e-4))
        disc_ids = set(id(p) for p in self.discriminator.parameters())
        embedder_params = [p for p in self.parameters() if id(p) not in disc_ids]
        embedder_opt = torch.optim.AdamW(embedder_params, lr=lr)
        disc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 0.5)
        return embedder_opt, disc_opt

    def custom_train_step(self, batch, optimizers, schedulers, batch_idx):
        self.train()
        embedder_opt, disc_opt = optimizers
        coords, msg, cover_img = batch['coords'], batch['msg'], batch['img']
        global_step = self.trainer.global_step

        # ---- Forward ----
        res = self(coords, msg, cover_img)
        wm_img = res['wm_img']
        predict_msg = res['predict_msg'][:, :self.decoder_mask_len]
        msg_trunc = msg[:, :self.decoder_mask_len]

        # ---- Msg Loss ----
        msg_loss = F.mse_loss(predict_msg, msg_trunc) * self.w_msg

        # ---- GAN factor ----
        gan_factor = adopt_weight(1.0, global_step, threshold=self.gan_start)

        # ============================================================
        # optimizer_idx=1: Discriminator update (pattern from videoseal)
        # ============================================================
        if gan_factor > 0:
            disc_opt.zero_grad()
            logits_real = self.discriminator(cover_img.detach())
            logits_fake = self.discriminator(wm_img.detach())
            disc_loss = gan_factor * hinge_d_loss(logits_real, logits_fake)
            self.loss_backward(disc_loss)
            disc_opt.step()
        else:
            disc_loss = torch.tensor(0.0, device=coords.device)

        # ============================================================
        # optimizer_idx=0: Embedder update (discriminator frozen)
        # ============================================================
        embedder_opt.zero_grad()

        # GAN loss: 冻结判别器
        if gan_factor > 0:
            requires_grad_orig = {}
            for p in self.discriminator.parameters():
                requires_grad_orig[p] = p.requires_grad
                p.requires_grad = False
            logits_fake = self.discriminator(wm_img)
            gan_loss = -logits_fake.mean()
            for p, val in requires_grad_orig.items():
                p.requires_grad = val
        else:
            gan_loss = torch.tensor(0.0, device=coords.device)

        msg_loss = F.mse_loss(predict_msg, msg_trunc) * self.w_msg

        # 不带梯度平衡，GAN 真正参与优化
        embedder_loss = msg_loss + self.w_gan * gan_factor * gan_loss
        self.loss_backward(embedder_loss)
        embedder_opt.step()

        # ---- Metrics ----
        from FastTools.steganography.utils.common import msg_acc
        acc = msg_acc(predict_msg, msg_trunc)
        psnr_val = -10 * torch.log10(F.mse_loss(wm_img.detach(), cover_img) + 1e-8)

        self.log('psnr', psnr_val.item(), prog_bar=True)
        self.log('loss', embedder_loss.item(), prog_bar=True)
        self.log('acc', acc.item(), prog_bar=True)
        self.log('msg_loss', msg_loss.item())
        self.log('gan_loss', gan_loss.item() if isinstance(gan_loss, torch.Tensor) else 0.0)
        self.log('disc_loss', disc_loss.item() if isinstance(disc_loss, torch.Tensor) else 0.0)
        self.log('gan_factor', gan_factor, prog_bar=True)
        self.log('msg_len', self.decoder_mask_len, prog_bar=True)

        if batch_idx == 0:
            noised_img = res.get('noised_img', wm_img)
            self.log_img("cover", cover_img.detach().cpu(), n_epoch=1)
            self.log_img("wm_clean", wm_img.detach().cpu(), n_epoch=1)
            self.log_img("wm_noised", noised_img.detach().cpu(), n_epoch=1)

        if batch_idx % 50 == 0:
            print(f'[step {batch_idx}] acc={acc.item():.4f} msg={msg_loss.item():.4f} gan={gan_loss.item() if isinstance(gan_loss, torch.Tensor) else 0:.4f} disc={disc_loss.item() if isinstance(disc_loss, torch.Tensor) else 0:.4f} psnr={psnr_val.item():.1f} gF={gan_factor:.3f}', flush=True)
