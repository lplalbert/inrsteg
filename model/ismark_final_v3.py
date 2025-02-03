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

from model.ResMLP import ResMLP
from model.Siren import FCBlock
from stable_signature.utils_model import get_hidden_decoder, get_hidden_decoder_ckpt

    
from FastTools.light.LightModel import LModel, LTrainer
from FastTools.util.TrainUtil import Args
from dataset.Mydataset import MyDataset, generate_grid_coordinates
from lightning.pytorch.callbacks import ModelCheckpoint


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class FourierFeatMapping(nn.Module):
    def __init__(self, in_dim, map_scale=16, map_size=512, tunable=False):
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


# Fourier feature mapping
class GaussianFourierFeatures(nn.Module):
    def __init__(self, num_input_channels, mapping_size=256, scale=10, tunable=False):
        super().__init__()
        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale
        if tunable:
            self._B = nn.Parameter(self._B, requires_grad=True)

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels, \
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)
        # print(x.shape, self._B.shape)
        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)




class FeatureGrid(nn.Module):
    def __init__(self, img_size, feat_dim=64, level=4, init_mode='none'):
        super().__init__()
        self.grids = nn.ParameterList([nn.Parameter(torch.randn(1, feat_dim, img_size // 2**i, img_size // 2**i), requires_grad=True) for i in range(0, level)])
        self.coords = nn.ParameterList([generate_grid_coordinates((-1, -1), 2 , img_size // 2**i).permute(2, 0, 1) for i in range(0, level)])
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
        for feat_grid, feat_coord in zip(self.grids, self.coords):
            
            feat_grid = feat_grid.expand(coords.shape[0], -1, -1, -1)
            feat = F.grid_sample(feat_grid, coords.flip(-1), align_corners=True, mode='nearest')
            feat = rearrange(feat, 'b c h w -> b (h w) c').contiguous()
            
            feat_coord = feat_coord.expand(coords.shape[0], -1, -1, -1)
            sample_coord = F.grid_sample(feat_coord, coords.flip(-1), align_corners=True, mode='nearest')
            sample_coord = rearrange(sample_coord, 'b c h w -> b (h w) c').contiguous()
            sample_coord = vector_coords - sample_coord
            # print(feat.size())
            feats.append(feat)
            sample_coords.append(sample_coord)
        return torch.cat(feats, dim=-1), torch.cat(sample_coords, dim=-1)        
        pass
    pass



class Decoder(nn.Module):
    '''
    Decode the encoded image and get message
    '''

    def __init__(self, message_length, in_channel=3, blocks=2, channels=64, diffusion_length=256, feature_size=64):
        super().__init__()
        self.diffusion_length = diffusion_length
        self.diffusion_size = int(self.diffusion_length ** 0.5)
        self.feature_size = feature_size
        self.first_layers = nn.Sequential(
            ConvBNRelu(in_channel, channels),
            SENet(channels, channels * 2, blocks=1),
            nn.MaxPool2d((2, 2)),
            ConvBNRelu(channels * 2, channels),
            SENet(channels, channels, blocks=1),
            nn.MaxPool2d((2, 2)),
            ConvBNRelu(channels, channels),
        )
        self.keep_layers = nn.Sequential(
            SENet(channels, channels, blocks=1),
            SENet_decoder(channels, channels * 2, blocks=blocks, drop_rate2=1),
            ConvBNRelu(channels * 2, channels),
            nn.AdaptiveAvgPool2d((self.diffusion_size, self.diffusion_size)),
        )

        self.final_layer = ConvBNRelu(channels, 1)

        self.message_layer = MLP(self.diffusion_length, 256, num_hidden_layers=2, norm='bn')
        # self.predictor = nn.ModuleList([MLP(256, 1, num_hidden_layers=1, norm='bn', act='relu') for _ in range(message_length)])
        self.predictor = MLP(256, message_length, num_hidden_layers=4, norm='bn', act='relu')


    def forward(self, noised_image):
        x = self.first_layers(noised_image)
        x = self.keep_layers(x)
        x = self.final_layer(x)
        z = x.view(x.shape[0], -1)
        dropout_z = F.dropout(z, p=0.1, training=self.training) # 加个dropout层
        z = self.message_layer(dropout_z)
        x = self.predictor(z)
        # x = torch.cat([predictor(z) for predictor in self.predictor], dim=-1)
        return x
    


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
        self.level_dim = 32
        self.level_num = 8
        self.diff_coords_dim = self.level_num * 2
        
        self.struct_embedding = FeatureGrid(self.img_size * 2, self.level_dim, level=self.level_num)
        
        # self.pe = FourierFeatMapping(2, 32, self.mapper_dim)
        self.msg_encoder = nn.Sequential(MLP(self.msg_len, self.feat_channel, num_hidden_layers=3, norm='bn', hidden_dim=256))
        
                
        # decoder = get_hidden_decoder(48, num_blocks=8)
        # ckpt = get_hidden_decoder_ckpt("/home/light_sun/workspace/inrsteg/stable_signature/hidden/ckpts/hidden_replicate.pth")
        # decoder.load_state_dict(ckpt)
        # if not self.tunable_decoder:
        #     for param in decoder.parameters():
        #         param.requires_grad = False
        #     decoder.eval()
        
        # self.decoder = nn.Sequential(
        #     decoder,
        #     MLP(48, self.msg_len, hidden_dim=256, num_hidden_layers=1, norm='none', act='lrelu'),
        # )
        
        self.decoder = get_hidden_decoder(self.msg_len, 3, 8)
        # 加了ln 貌似就会导致全都预测同一种颜色了
        # self.merge_layer = MLP(self.level_dim * self.level_num + self.diff_coords_dim, self.feat_channel, norm='none', act='relu', num_hidden_layers=6, hidden_dim=256)
        
        self.inr = ResMLP(self.level_dim * self.level_num + self.diff_coords_dim + self.feat_channel + 3, 3, num_layers=6, hidden_dim=256, norm='none', activation='relu')

        

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
            ("Quantize", None)
        ])
        
        pass
    
    def render_img(self, coords, msg, img):
        bs, h, w, c = coords.size()
        struct_feat, diff_coords_feat = self.struct_embedding(coords)
        msg_feat = self.msg_encoder(msg) # [bs, feat_channel]
        msg_feat = msg_feat.unsqueeze(1).expand(-1, struct_feat.size(1), -1)
        rgb = rearrange(img, 'b c h w -> b (h w) c')
        feat = torch.cat([struct_feat, diff_coords_feat, msg_feat, rgb], dim=-1)
        
        mask = self.inr(feat)
        wm_img = mask + rgb
        wm_img = rearrange(wm_img, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        mask = rearrange(mask, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        return wm_img, mask
        pass
    
    def decode_msg(self, img):
        return self.decoder(img)
        pass

    def forward(self, coords, msg, img):
        wm_img, mask = self.render_img(coords, msg, img)
        
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
              "mask": mask * 0.5 + 0.5
        }
        pass


    def custom_train_step(self, batch, optimizers, schedulers, batch_idx):
        self.train()
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


        loss = msg_loss + img_loss
        self.loss_backward(loss)
        optimizer.step()

        acc = msg_acc(predict_msg, msg)
        psnr = PSNR(wm_img, cover_img)
        
        self.log('psnr', psnr.cpu().item(), prog_bar=True)
        self.log('metric/img_loss', img_loss.cpu().item())
        self.log('metric/msg_loss', msg_loss.cpu().item())
                        
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
        
        loss = msg_loss + img_loss
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
        return MyDataset(cfg, data_len=1000000), MyDataset(cfg, data_len=1000)
    
    
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
    args = Args().load("/home/light_sun/workspace/inrsteg/config/main.yaml")
    
    model = INRMark(args)
    img_size = 128
    msg_len = 48
    model(
        torch.clamp(torch.rand(2, img_size, img_size, 2), -1, 1), 
        torch.rand(2, msg_len), 
        torch.clamp(torch.randn(2, 3, img_size, img_size), 0, 1))
    pass

