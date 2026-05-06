# Standard library imports
from typing import Optional, Tuple

# Third-party imports
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

# Local imports
from FastTools.light.Engine import EngineModel, EngineTrainer
from FastTools.light.LightModel import LModel
from FastTools.metre import PSNR
from FastTools.module.common import Conv2dBlock, ConvBNRelu, MLP
from FastTools.steganography.Noiser.Noiser import Noiser
from FastTools.steganography.utils.common import msg_acc
from FastTools.util.ImgUtil import clip_psnr
from FastTools.util.TrainUtil import Args
from dataset.Mydataset import MyDataset, generate_grid_coordinates

# 采用v4的权重，去掉没用的傅里叶卷积, 添加坐标映射, 激活函数变成GeLu
class FourierFeatMapping(nn.Module):
    # 基于理论Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
    def __init__(self, in_dim, map_scale=16, map_size=4, tunable=False):
        super().__init__()

        B = torch.normal(0., map_scale, size=(map_size//2, in_dim))

        if tunable:
            self.B = nn.Parameter(B, requires_grad=True)
        else:
            self.register_buffer('B', B)

    @property
    def out_dim(self):
        return 2 * self.B.shape[0]

    @property
    def flops(self):
        return self.B.shape[0] * self.B.shape[1]

    def forward(self, x):
        x_proj = torch.matmul(x, self.B.T)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
 
 
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.groups = groups

        # 调整通道数处理：输入输出通道数翻倍（实部+虚部）
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,  # 处理实部和虚部
            out_channels=out_channels * 2,
            kernel_size=1,
            groups=self.groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

        self.out_layer = nn.Sequential(
            nn.Conv2d(out_channels+in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        batch, c, h, w = x.shape

        # 新版FFT处理
        # 步骤1：执行FFT变换
        fft = torch.fft.rfft2(x, norm='ortho')  # 输出复数张量 [B,C,H,W//2+1]

        # 分离实部和虚部
        real = fft.real  # [B,C,H,W//2+1]
        imag = fft.imag  # [B,C,H,W//2+1]

        # 拼接实部虚部作为通道维度
        fft_combined = torch.cat([real, imag], dim=1)  # [B, 2*C, H, W//2+1]

        # 步骤2：频域卷积处理
        fft_processed = self.conv_layer(
            fft_combined)  # [B, 2*out_C, H, W//2+1]
        fft_processed = self.relu(self.bn(fft_processed))

        # 拆分处理后的实部虚部
        real_new, imag_new = torch.chunk(
            fft_processed, 2, dim=1)  # 各[B, out_C, H, W//2+1]

        # 步骤3：逆FFT变换
        fft_new = torch.complex(real_new, imag_new)  # 重建复数张量
        output = torch.fft.irfft2(fft_new, s=(
            h, w), norm='ortho')  # [B, out_C, H, W]
        output = self.out_layer(torch.cat([output, x], dim=1))
        return output


class LowRankFusionBlock(nn.Module):
    def __init__(self, x_dim, y_dim, rank_dim=16, out_dim=None):
        super().__init__()
        self.x_dim = x_dim + 1
        self.y_dim = y_dim + 1
        self.rank_dim = rank_dim
        if out_dim is None:
            self.out_dim = x_dim
        else:
            self.out_dim = out_dim
        self.x_fc = nn.Linear(self.x_dim, self.rank_dim, bias=False)
        self.y_fc = nn.Linear(self.y_dim, self.rank_dim, bias=False)
        self.out_fc = nn.Linear(self.rank_dim, self.out_dim, bias=False)
        self.act = nn.ReLU(True)
        self.bn = nn.BatchNorm1d(self.out_dim)
        self.k = nn.Linear(
            x_dim, self.out_dim, bias=False) if self.out_dim != x_dim else nn.Identity()
        pass

    def forward(self, x, y):
        bs = x.size(0)
        raw_x = x
        x = torch.cat([x, torch.ones(bs, 1, device=x.device)], dim=1)
        y = torch.cat([y, torch.ones(bs, 1, device=y.device)], dim=1)
        x = self.x_fc(x)
        y = self.y_fc(y)
        z = x * y
        z = self.out_fc(z)
        z = self.bn(z)
        z = self.act(z)
        return self.k(raw_x) + z
    pass


class HiddenDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """

    def __init__(self, msg_len=30, hidden_channels=128, blocks=4, repeat=3):

        super(HiddenDecoder, self).__init__()
        self.channels = hidden_channels

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))
            layers.append(
                nn.Sequential(
                    # FourierUnit(self.channels, self.channels),
                    SEBlock(self.channels),
                    Conv2dBlock(self.channels, self.channels, 4, 2, 0, norm='bn', activation='relu')))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(self.channels, msg_len * repeat))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(msg_len * repeat, msg_len)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        # the output is of shape b x c x 1 x 1, and we want to squeeze out the last two dummy dimensions and make
        # the tensor of shape b x c. If we just call squeeze_() it will also squeeze the batch dimension when b=1.
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)
        return x


class FeatureGrid(nn.Module):
    """Learnable feature grid with multi-resolution levels.

    Args:
        img_size: Base image size
        feat_dim: Feature dimension per level
        level: Number of grid levels
        init_mode: Weight initialization mode
        sample_mode: Feature sampling mode
    """

    def __init__(self,
                 img_size: int,
                 feat_dim: int = 16,
                 level: int = 8,
                 init_mode: str = 'none',
                 sample_mode: str = 'bilinear',
                 out_dim: int = 128,
                 addition_fourier_feat=True,
                 fourier_feat_dim=16,
                 fourier_feat_tunable=True):
        super().__init__()
        self.sample_mode = sample_mode

        # Create multi-resolution grids
        self.grids = nn.ParameterList([
            nn.Parameter(torch.randn(
                1, feat_dim, img_size // 2**i, img_size // 2**i))
            for i in range(level)
        ])

        self.out = nn.Sequential(
            nn.Linear(feat_dim * level + fourier_feat_dim, out_dim),
            # nn.LayerNorm(out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
        ) if out_dim is not None else nn.Identity()
        if addition_fourier_feat:
            self.adapter = nn.Sequential(
                nn.Linear(feat_dim * level + fourier_feat_dim, feat_dim * level + fourier_feat_dim),
                nn.BatchNorm1d(feat_dim * level + fourier_feat_dim),
                nn.SiLU()
            )
        else:
            self.adapter = nn.Identity()
        self.addition_fourier_feat = addition_fourier_feat
        
        if addition_fourier_feat:
            self.fourier_mapper = FourierFeatMapping(
                2,  
                map_size=fourier_feat_dim, 
                tunable=fourier_feat_tunable)

        # Initialize weights
        # if init_mode == 'sine':
        #     for grid in self.grids:
        #         num_input = grid.data.size(-1)
        #         grid.data.uniform_(-np.sqrt(6 / num_input) /
        #                            30, np.sqrt(6 / num_input) / 30)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Sample features from multi-resolution grids.

        Args:
            coords: Input coordinates [B, H, W, 2]

        Returns:
            Concatenated features from all levels
        """
        batch_size, height, width, _ = coords.shape

        feats = []
        for grid in self.grids:
            # Expand grid to batch size and sample features
            grid = grid.expand(batch_size, -1, -1, -1)
            feat = F.grid_sample(grid, coords.flip(-1),
                                 align_corners=True, mode=self.sample_mode)
            feat = rearrange(feat, 'b c h w -> b (h w) c')
            feats.append(feat)
            
        if self.addition_fourier_feat:
                fourier_feat = self.fourier_mapper(coords).permute(0, 3, 1, 2)
                fourier_feat = rearrange(fourier_feat, 'b c h w -> b (h w) c')
                feats.append(fourier_feat)
                pass
                  
        feats = torch.cat(feats, dim=-1)
        # 调整形状
        b,n,c = feats.size()
        feats = rearrange(feats, 'b n c -> (b n) c')
        feats = self.adapter(feats)
        feats = self.out(feats)
        feats = rearrange(feats, '(b n) c -> b n c',b=b,n=n)
        return feats


class INRMark(EngineModel):
    """INR-based Image Watermarking Model.

    Args:
        args: Configuration parameters
    """

    def __init__(self, args: Args):
        super().__init__(args)
        # global parameters
        self.noised = args.noised
        self.fixed_psnr = args.fixed_psnr
        # Architecture parameters
        self.img_size = args.img_size
        self.msg_len = args.msg_len
        self.level_dim = 32
        self.level_num = 0
        self.msg_dim = 128
        self.rank_dim = 128
        self.alpha = args.alpha
        # Loss weights
        self.w_msg = args.w_msg
        self.w_img = args.w_img
        self.w_lpips = args.w_lpips
        self.struct_dim = 256

        # Components
        self.struct_embedding = FeatureGrid(
            img_size=self.img_size*2,
            feat_dim=self.level_dim,
            level=self.level_num,
            sample_mode='bilinear',
            out_dim=self.struct_dim)

        # Message transformation layers
        # self.S = nn.Linear(self.struct_dim, self.msg_len)
        # self.D = nn.Linear(self.msg_len, self.struct_dim)

        # Networks
        self.msg_encoder = MLP(
            in_dim=self.msg_len,
            out_dim=self.msg_dim,
            num_hidden_layers=2,
            norm='bn'
        )

        self.inr = nn.ModuleList([
            LowRankFusionBlock(self.struct_dim, self.msg_dim, self.rank_dim),
            LowRankFusionBlock(self.struct_dim, self.msg_dim, self.rank_dim),
            LowRankFusionBlock(self.struct_dim, self.msg_dim, self.rank_dim),
            LowRankFusionBlock(self.struct_dim, self.msg_dim, self.rank_dim)
        ])
        self.predict = nn.Linear(self.struct_dim, 3)

        self.decoder = HiddenDecoder(self.msg_len)

        # self.inr = MLP(
        #     in_dim=self.struct_dim,
        #     out_dim=3,
        #     num_hidden_layers=2,
        #     hidden_dim=256,
        #     norm='bn',
        #     act='relu'
        # )

        # Augmentation
        self.noiser = Noiser([
            ("Identity", None),
            ("Rotate", None),
            ("Crop", None),
            ("Translate", None),
            ("Scale", None),
            ("Shear", None),
            ("Dropout", None),
            ("Cropout", None),

            ("Color", None),
            ("KorniaJpeg", None),
            ("GaussianFilter", None),
            ("GaussianNoise", None),

        ])

        # Metrics
        # self.lpips = LPIPS(net='vgg')

    def decode_msg(self, x):
        return self.decoder(x)
    
    def render_img(self,
                   coords: torch.Tensor,
                   msg: torch.Tensor,
                   img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Render watermarked image.

        Args:
            coords: Coordinate grid [B, H, W, 2]
            msg: Secret message [B, msg_len]
            img: Original image [B, 3, H, W]

        Returns:
            wm_img: Watermarked image
            mask: Watermark mask
        """
        bs, h, w, c = coords.size()
        n = h * w
        # Sample structural features
        struct_feat = self.struct_embedding(coords)  # [b n c]
        struct_feat = rearrange(struct_feat, "b n c -> (b n) c")

        # Message transformation
        msg_feat = self.msg_encoder(msg)
        msg_feat = msg_feat.unsqueeze(1).repeat(1, n, 1)  # [b n c]
        msg_feat = rearrange(msg_feat, "b n c -> (b n) c")
        for layer in self.inr:
            struct_feat = layer(struct_feat, msg_feat)
            pass
        # Generate watermark mask
        residual = self.predict(struct_feat)
        residual = rearrange(residual, "(b n) c -> b n c", b=bs, n=n)
        residual = rearrange(residual, "b (h w) c -> b c h w", h=h, w=w)
        residual = torch.clamp(residual, -1, 1)

        # Apply watermark
        wm_img = residual * self.alpha + img  # Adjust alpha as needed
        return torch.clamp(wm_img, 0, 1), residual

    def forward(self,
                coords: torch.Tensor,
                msg: torch.Tensor,
                img: torch.Tensor) -> dict:
        # Generate watermarked image
        wm_img, mask = self.render_img(coords, msg, img)

        # Apply PSNR constraint
        if self.fixed_psnr:
            wm_img = clip_psnr(wm_img, img, psnr=self.fixed_psnr)

        # Apply augmentations
        noised_img = self.noiser(wm_img, img)[0] if self.noised else wm_img

        # Decode message
        pred_msg = self.decoder(noised_img)

        return {
            "img": img,
            "predict_msg": pred_msg,
            "wm_img": wm_img,
            "noised_img": noised_img,
            "mask": mask
        }

    def custom_train_step(self, batch, optimizers, schedulers, batch_idx):
        self.train()
        # self.decoder[1].eval()
        optimizer: torch.optim.Adam = optimizers
        optimizer.zero_grad()
        coords = batch['coords']
        msg = batch['msg']
        cover_img = batch['img']

        res = self(coords, msg, cover_img)

        wm_img = res['wm_img']
        predict_msg = res['predict_msg']
        noised_img = res['noised_img']
        msg_loss = F.mse_loss(predict_msg, msg) * self.w_msg

        img_loss = F.mse_loss(wm_img, cover_img) * self.w_img
        # lpips_loss = self.w_lpips * torch.mean(self.lpips.forward(wm_img*2-1,cover_img*2-1))

        loss = msg_loss + img_loss  # + lpips_loss
        self.loss_backward(loss)
        optimizer.step()

        acc = msg_acc(predict_msg, msg)
        psnr = PSNR(wm_img, cover_img)

        self.log('psnr', psnr.cpu().item(), prog_bar=True)
        self.log('metric/img_loss', img_loss.cpu().item())
        self.log('metric/msg_loss', msg_loss.cpu().item())
        # self.log('metric/lpips_loss', lpips_loss.cpu().item())

        self.log('loss', loss.cpu().item(), prog_bar=True)
        self.log('acc', acc.cpu().item(), prog_bar=True)

        if batch_idx == 0:
            self.log_img("img", cover_img.detach().cpu(), n_epoch=1)
            self.log_img("wm_img", wm_img.detach().cpu(), n_epoch=1)
            self.log_img("mask", res['mask'].detach().cpu(), n_epoch=1)
            self.log_img("noised_img", noised_img.detach().cpu(), n_epoch=1)
        pass

    def custom_valid_step(self, batch, batch_idx):
        self.eval()
        coords = batch['coords']
        msg = batch['msg']
        cover_img = batch['img']

        res = self(coords, msg, cover_img)
        wm_img = res['wm_img']
        predict_msg = res['predict_msg']

        msg_loss = F.mse_loss(predict_msg, msg) * self.w_msg
        img_loss = F.mse_loss(wm_img, cover_img) * self.w_img
        # lpips_loss = self.w_lpips * torch.mean(self.lpips.forward(wm_img*2-1,cover_img*2-1))

        loss = msg_loss + img_loss  # + lpips_loss
        acc = msg_acc(predict_msg, msg)
        psnr = PSNR(wm_img, cover_img)

        self.log("metric_val/psnr", psnr.cpu().item(), sync_dist=True)
        self.log('val_loss', loss.cpu().item(), sync_dist=True)
        self.log('metric_val/val_acc', acc.cpu().item(), sync_dist=True)
        pass

    def build_optimizers(self, args):
        return torch.optim.AdamW(self.parameters(), lr=4e-4)
        pass


class INRMarkTrainer(EngineTrainer):

    def build_dataset(self, cfg):
        return MyDataset(cfg, data_len=50000), MyDataset(cfg, data_len=100)

    def build_model(self, cfg):
        return INRMark(cfg)

    def build_checkpoint_callback(self):

        callback = ModelCheckpoint(
            save_top_k=5,  # 默认保存最好的5个， 需要保存条件
            monitor='val_loss',  # 默认使用总loss保存最好的结果
            filename="ckpt-{epoch:02d}-{val_loss:.4f}",
            save_last=True,
            every_n_epochs=1,
            # save_on_train_epoch_end=True,
            save_weights_only=False
        )

        return callback

    pass


if __name__ == '__main__':
    args = Args().load("/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/main.yaml")

    model = INRMark(args)
    img_size = 128
    msg_len = 30
    model(
        torch.clamp(torch.rand(2, img_size, img_size, 2), -1, 1),
        torch.rand(2, msg_len),
        torch.clamp(torch.randn(2, 3, img_size, img_size), 0, 1))
    pass
