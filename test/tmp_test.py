from einops import rearrange
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


def generate_grid_coordinates(top_left, side_length, grid_size=128):
    """
    生成固定大小的坐标矩阵
    
    :param top_left: 左上角点的坐标 (x1, y1)
    :param side_length: 矩阵的边长
    :param grid_size: 坐标矩阵的大小 (默认 128x128)
    :return: 坐标矩阵 [grid_size, grid_size, 2]
    """
    # 生成网格坐标
    x = torch.linspace(top_left[0], top_left[0] + side_length, grid_size)
    y = torch.linspace(top_left[1], top_left[1] + side_length, grid_size)
    
    # 生成坐标矩阵
    xv, yv = torch.meshgrid(x, y, indexing='ij')
    coordinates = torch.stack([xv, yv], dim=-1)
    
    return coordinates

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

if __name__ == "__main__":
    size = 10
    coords = generate_grid_coordinates((-1, -1), 2, 128).unsqueeze(0)
    model = FeatureGrid(256, level=4)
    print(coords.shape)
    print(coords)
    feat, coords = model(coords)
    print(coords.size())
    print(coords.max(), coords.min())
    pass