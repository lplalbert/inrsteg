from einops import rearrange
import torch.nn as nn
import torch
import torch.nn.functional as F
from FastTools.light.Engine import EngineModel, EngineTrainer
from FastTools.light.LightModel import LModel
from FastTools.metre import PSNR
from FastTools.module.SENet import SENet, SENet_decoder
from FastTools.module.common import MLP, ConvBNRelu, LinearBlock, ResBlock
from FastTools.steganography.Noiser.Noiser import Noiser
from FastTools.steganography.utils.common import msg_acc
from FastTools.util.ImgUtil import clip_psnr
from FastTools.util.TrainUtil import Args
from FastTools.util.utils import weights_init
import numpy as np
import lpips
from model.AFFormer import AFFormerDecoder
from model.ResMLP import ResMLP
from model.Siren import FCBlock
from model.inr_base import PosEncodingNeRF
from stable_signature.utils_model import get_hidden_decoder, get_hidden_decoder_ckpt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
    
from FastTools.light.LightModel import LModel, LTrainer
from FastTools.util.TrainUtil import Args
from dataset.Mydataset import MyDataset, generate_grid_coordinates
from lightning.pytorch.callbacks import ModelCheckpoint
from stlpips_pytorch import stlpips
import torchmetrics

# 只用feature grid
# 将msg改成

class FeatureGrid(nn.Module):
    def __init__(self, img_size, feat_dim=64, level=4, init_mode='none', sample_mode='nearest', use_diff_coord=True):
        super().__init__()
        self.sample_mode = sample_mode
        self.use_diff_coord = use_diff_coord
        self.grids = nn.ParameterList([nn.Parameter(torch.randn(1, feat_dim, img_size // 2**i, img_size // 2**i), requires_grad=True) for i in range(0, level)])
        if use_diff_coord:
            self.coords = nn.ParameterList([generate_grid_coordinates((-1, -1), 2 , img_size // 2**i).permute(2, 0, 1) for i in range(0, level)])
        else:
            self.coords = None
        # 初始化
        if init_mode == 'sine':
            for grid in self.grids:
                num_input = grid.data.size(-1)
                grid.data.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
                
        pass
    
    def forward(self, coords):
        """从Feature Grid中采样特征

        Args:
            coords (_type_): _description_
        """
        vector_coords = rearrange(coords, "b h w c -> b (h w) c")
        
        feats = []
        sample_coords = []
        if self.use_diff_coord:
            for feat_grid, feat_coord in zip(self.grids, self.coords):
                # 感觉feat还是得是最近的，不能是线性插值，不然我的diff coords就没意义了
                feat_grid = feat_grid.expand(coords.shape[0], -1, -1, -1)
                feat = F.grid_sample(feat_grid, coords.flip(-1), align_corners=True, mode=self.sample_mode)
                feat = rearrange(feat, 'b c h w -> b (h w) c').contiguous()
                feats.append(feat)
                feat_coord = feat_coord.expand(coords.shape[0], -1, -1, -1)
                sample_coord = F.grid_sample(feat_coord, coords.flip(-1), align_corners=True, mode='nearest')
                sample_coord = rearrange(sample_coord, 'b c h w -> b (h w) c').contiguous()
                sample_coord = vector_coords - sample_coord
                sample_coords.append(sample_coord)
        else:
            for feat_grid in self.grids:
                # 感觉feat还是得是最近的，不能是线性插值，不然我的diff coords就没意义了
                feat_grid = feat_grid.expand(coords.shape[0], -1, -1, -1)
                feat = F.grid_sample(feat_grid, coords.flip(-1), align_corners=True, mode=self.sample_mode)
                feat = rearrange(feat, 'b c h w -> b (h w) c').contiguous()
                feats.append(feat)
        if self.use_diff_coord:
            return torch.cat(feats, dim=-1), torch.cat(sample_coords, dim=-1)        
        else:
            return torch.cat(feats, dim=-1)
        pass
    pass



   
class ImageNormalizer(nn.Module):
    def __init__(self):
        super(ImageNormalizer, self).__init__()
        self.image_mean = torch.Tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.image_std = torch.Tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def forward(self, x):
        """ Normalize image to approx. [-1,1] """
        return (x - self.image_mean.to(x.device)) / self.image_std.to(x.device)

class INRMark(EngineModel):
    def __init__(self, args):
        super(INRMark, self).__init__(args)
        self.img_size = args.img_size
        self.feat_channel = args.feat_dim
        self.msg_len = args.msg_len
        self.decoder_blocks = args.decoder_blocks
        self.use_bg = True
        self.use_fold = args.use_fold
        self.mapper_dim = 256
        self.tunable_decoder = args.tunable_decoder
        self.fixed_psnr = args.fixed_psnr
        self.inject_rgb = args.inject_rgb
        self.noised = args.noised
        self.w_msg = args.w_msg
        self.w_img = args.w_img
        self.w_lpips = args.w_lpips
        self.level_dim = 64
        self.level_num = 8
        self.diff_coords_dim = self.level_num * 2
        self.use_cell = args.use_cell
        
        self.struct_embedding = FeatureGrid(self.img_size * 2, self.level_dim, level=self.level_num, sample_mode='bilinear', use_diff_coord=False)
        
        # self.coord_pe = nn.Sequential(
        # #     PosEncodingNeRF(
        # #     in_features=2,
        # #     num_frequencies=64,
        # #     scale=2.
        # # ),
        #     MLP(2, self.feat_channel, 256, 2, 'none')
        # )
        # self.pe_dim = 64 * 4 + 2
        
        self.S = nn.Linear(self.level_dim * self.level_num, self.msg_len)
        self.D = nn.Linear(self.msg_len, self.level_dim * self.level_num)
        
        self.msg_encoder = MLP(self.msg_len, self.msg_len * self.msg_len, num_hidden_layers=4, norm='bn', hidden_dim=256)
            

        # self.decoder = AFFormerDecoder(self.msg_len, 1)
        self.decoder = get_hidden_decoder(self.msg_len, 3, 8)
        self.inr = MLP(self.level_dim * self.level_num, 3, num_hidden_layers=4, hidden_dim=512, norm='bn', act='relu')


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
        
        pass
    
    def render_img(self, coords, msg, img, cell=None):
        bs, h, w, c = coords.size()
        n = h * w
        
        # 潜在结构特征
        struct_feat = self.struct_embedding(coords)
        struct_feat = rearrange(struct_feat, "b n c -> (b n) c")
        
        # 坐标编码
        # vec_coords = rearrange(coords, "b h w c -> (b h w) c")
        # coords_feat = self.coord_pe(vec_coords)
        
        # 消息编码
        msg_feat = self.msg_encoder(msg) # [bs, dim]
        msg_feat = rearrange(msg_feat, "b (h w) -> b 1 h w", h=self.msg_len, w=self.msg_len)
        msg_feat = msg_feat.expand(-1, n, -1, -1) # [bs, n, dim]
        msg_feat = rearrange(msg_feat, "b n h w -> (b n) h w")
        
        tmp_feat = self.S(struct_feat).unsqueeze(1)
        tmp_feat = (tmp_feat @ msg_feat).squeeze(1)
        tmp_feat = self.D(tmp_feat)
        
        
        # 含有水印信息的结构特征
        wm_struct_feat = struct_feat + tmp_feat
        
        # 解码mask
        # feat = torch.cat([struct_feat, msg_feat], dim=-1)

        mask = self.inr(wm_struct_feat)
        mask = rearrange(mask, "(b n) d -> b n d", b=bs, n=n).contiguous()
        mask = rearrange(mask, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
    
        
        wm_img = mask + img
        wm_img = torch.clamp(wm_img, 0, 1)
        
        return wm_img, torch.clamp(mask*0.5+0.5, 0, 1)
        pass
    
    def decode_msg(self, img):
        return self.decoder(img)
        pass

    def forward(self, coords, msg, img, cell=None):
        wm_img, mask = self.render_img(coords, msg, img, cell)
        
        if self.fixed_psnr:
            wm_img = clip_psnr(wm_img, img, psnr=self.fixed_psnr)
            
        if self.noised:
            noised_img, _ = self.noiser(wm_img, img)
        else:
            noised_img = wm_img

        predict_msg = self.decoder(noised_img)
        
        return {
              "img": img,
              "predict_msg": predict_msg,
              "noised_img": noised_img,
              "wm_img": wm_img,
              "mask": mask
        }
        pass


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


        loss = msg_loss + img_loss # + lpips_loss
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

        loss = msg_loss + img_loss # + lpips_loss
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
        return MyDataset(cfg, data_len=10000), MyDataset(cfg, data_len=100)
    
    
    def build_model(self, cfg):
        return INRMark(cfg)

    def build_checkpoint_callback(self):
        
        callback = ModelCheckpoint(
            save_top_k= 5, # 默认保存最好的5个， 需要保存条件
            monitor='val_loss', # 默认使用总loss保存最好的结果
            filename="ckpt-{epoch:02d}-{val_loss:.4f}",
            save_last=True,
            every_n_epochs=5,
            # save_on_train_epoch_end=True,
            save_weights_only=False
        )
        
        return callback
    
    pass
    

if __name__ == "__main__":
    args = Args().load("/home/sn/workspace/inrsteg/config/main.yaml")
    
    model = INRMark(args)
    img_size = 128
    msg_len = 30
    model(
        torch.clamp(torch.rand(2, img_size, img_size, 2), -1, 1), 
        torch.rand(2, msg_len), 
        torch.clamp(torch.randn(2, 3, img_size, img_size), 0, 1))
    pass

