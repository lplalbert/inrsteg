


import sys
# from liif.utils import make_coord
from liif.utils import make_coord
import torch


def gen_coords_grid(top_left, side_length, grid_size=128, flatten=False):
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
    if flatten:
        coordinates = coordinates.view(-1, coordinates.shape[-1])
    return coordinates


coords = make_coord((100, 100), [[-0.5, 0.5], [-0.5, 0.5]], flatten=False)
print(coords, coords.size())
print(gen_coords_grid((-1, -1), 2, grid_size=100, flatten=True).size())