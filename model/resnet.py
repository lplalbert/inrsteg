from einops import rearrange
import torch

from torchvision import models
import os
import requests
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

model_ckpt = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
cache_path = "/home/light_sun/workspace/inrsteg/output/ckpt"


def download_file(url, file_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with open(file_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))
            
def fetch_model_weights(model_name):
    file_path = os.path.join(cache_path, model_ckpt[model_name].split('/')[-1])
    if not os.path.exists(file_path):
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        download_file(model_ckpt[model_name], file_path)
    return torch.load(file_path, map_location='cpu')



class ImgGrid(nn.Module):
    def __init__(self, img_size, out_dim=64, backbone='resnet18', tunable=False):
        super().__init__()
        model = models.__dict__[backbone](pretrained=False)
        model.load_state_dict(fetch_model_weights(backbone))
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.model = nn.ModuleList([
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ])
        self.out_dim = out_dim
        self.img_size = img_size
        if backbone == 'resnet18':
            self.hidden_dim = 64
        elif backbone == 'resnet50':
            self.hidden_dim = 256
        else:
            self.hidden_dim = 64

        self.projectors = nn.ModuleList([
            nn.Conv2d(self.hidden_dim, self.out_dim, kernel_size=1),
            nn.Conv2d(2 * self.hidden_dim, self.out_dim, kernel_size=1),
            nn.Conv2d(4 * self.hidden_dim, self.out_dim, kernel_size=1),
            nn.Conv2d(8 * self.hidden_dim, self.out_dim, kernel_size=1)
        ])
        
        if not tunable:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        
        
        pass
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feats = []
        for layer, projector in zip(self.model, self.projectors):
            x = layer(x)
            feat = projector(x)
            feat = F.interpolate(feat, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            feat = rearrange(feat, 'b c h w -> b (h w) c').contiguous()
            feats.append(feat)
            
        
        return torch.cat(feats, dim=-1)
    pass


if __name__ == "__main__":
    model = ImgGrid(backbone='resnet50')
    x = torch.randn(2, 3, 128, 128)
    coords = torch.nn.functional.tanh(torch.randn(2, 128, 128, 2))
    feats = model(x, coords)
    print(feats.size())
    pass