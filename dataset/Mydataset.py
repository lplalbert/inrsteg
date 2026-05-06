
from random import random
from random import choice
from FastTools.dataset.dataset import LDataset, read_img
import torch

from FastTools.steganography.utils.common import gen_random_msg
from FastTools.util.TrainUtil import Args
import os
from torchvision import transforms

from FastTools.util.utils import vutils
import numpy as np
import torch.nn.functional as F

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def gen_start_coords(min_scale=0):
    min_scale *= 2
    width = np.random.uniform(min_scale, 2)
    start_x = np.random.uniform(-1, 1 - width)
    start_y = np.random.uniform(-1, 1 - width)
    return (start_x, start_y), width
    pass

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

class MyDataset(LDataset):
    def __init__(self, args, data_len=1000000, valid=False) -> None:
        super().__init__(args)
        self.use_bg = args.use_bg
        self.num_data = data_len
        self.img_size = args.img_size
        self.msg_len = args.msg_len
        self.use_global = args.use_global
        self.use_cell = args.use_cell
        if valid:
            self.data_path = args.valid_data_path or "/home/light_sun/workspace/inrsteg/data/DIV2K_valid"
        else:
            self.data_path = args.train_data_path or "/home/light_sun/workspace/inrsteg/data/DIV2K_train"
        self.files = [
            name for name in os.listdir(self.data_path)
            if name.lower().endswith(IMG_EXTENSIONS)
        ]
        self.files.sort()
        if not self.files:
            raise FileNotFoundError("No image files found in dataset path: {}".format(self.data_path))
        self.cache = args.cache     
        self.global_ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.06, 1))
        ])
        self.min_scale = args.min_scale
        self.memory = []
        if args.cache:
            self.memory = [read_img(os.path.join(self.data_path, name)) for name in self.files]
        pass
    def __getitem__(self, index):
        # 在img_size大小下随机采样四个顶点坐标
        # 从[-1, 1]之间随机采样一个点
        
        # x = random() * (1-self.min_scale) 
        # y = random() * (1-self.min_scale) 

        # max_side = min(1-x, 1-y)
        # side_length = 2 * (random() * (max_side - self.min_scale) + self.min_scale)
        # x = x * 2 - 1
        # y = y * 2 - 1
        
        (x, y), side_length = gen_start_coords(self.min_scale)
        
        grids = generate_grid_coordinates((x, y), side_length, grid_size=self.img_size)
        msg = gen_random_msg(self.msg_len)
        ret = {
            "coords": grids,
            "msg": msg
        }

        if self.use_cell:
            ret['cell'] = torch.Tensor([side_length / 2]).float()
            
        if self.use_bg:
            if self.cache:
                img = choice(self.memory)
            else:
                img = read_img(os.path.join(self.data_path, choice(self.files)))
            global_img = self.global_ts(img)
            img = global_img # F.grid_sample(global_img.unsqueeze(0), grids.unsqueeze(0).flip(-1), mode='bilinear', align_corners=True).squeeze(0)
            ret['img'] = torch.clamp(img, 0, 1)
            if self.use_global:
                global_img = F.interpolate(global_img.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=True).squeeze(0)
                ret['global_img'] = torch.clamp(global_img, 0, 1)
        
        return ret
    
    def __len__(self):
        return self.num_data
        pass



if __name__ == "__main__":
    cfg = Args().load("/home/sn/workspace/inrsteg/config/main.yaml")
    dataset = MyDataset(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:

        msg = batch['msg']
        img = batch['img']
        coords = batch['coords']
        cell = batch['cell']
        print(cell.size())
        print(img.size())
        print(coords.size())
        print(cell)
        break

    pass
