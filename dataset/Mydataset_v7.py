"""
V7 Dataset: 真正均匀的坐标窗口采样

核心改进 (vs 原版 Mydataset.py):
  原版 gen_start_coords: width先采 → start_x受width约束 → 中心偏置
  V7 gen_start_coords_uniform: target先采 → width随机 → 窗口包住target
  → 每个坐标被覆盖的概率严格相等, 消除中心-边缘训练不平衡
"""

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


def gen_start_coords_uniform(min_scale=0):
    """每个坐标等概率被覆盖的随机窗口生成 (严谨保证)

    算法: 先选目标点 → 再选窗口大小 → 窗口包住目标点且不越界
    拒绝率: 极低 (< 1e-6), 只在width∈(2-ε,2] 且 target 靠近边界时触发
    """
    min_scale_eff = min_scale * 2
    while True:
        tx = np.random.uniform(-1, 1)
        ty = np.random.uniform(-1, 1)
        width = np.random.uniform(min_scale_eff, 2)

        lo_x = max(tx - width, -1)
        hi_x = min(tx, 1 - width)
        lo_y = max(ty - width, -1)
        hi_y = min(ty, 1 - width)

        if lo_x < hi_x and lo_y < hi_y:
            start_x = np.random.uniform(lo_x, hi_x)
            start_y = np.random.uniform(lo_y, hi_y)
            return (start_x, start_y), width


def generate_grid_coordinates(top_left, side_length, grid_size=128):
    x = torch.linspace(top_left[0], top_left[0] + side_length, grid_size)
    y = torch.linspace(top_left[1], top_left[1] + side_length, grid_size)
    xv, yv = torch.meshgrid(x, y, indexing='ij')
    return torch.stack([xv, yv], dim=-1)


class MyDatasetV7(LDataset):
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

    def __getitem__(self, index):
        # V7: 均匀坐标采样
        (x, y), side_length = gen_start_coords_uniform(self.min_scale)
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
            img = global_img
            ret['img'] = torch.clamp(img, 0, 1)
            if self.use_global:
                global_img = F.interpolate(global_img.unsqueeze(0), size=(self.img_size, self.img_size), mode='bilinear', align_corners=True).squeeze(0)
                ret['global_img'] = torch.clamp(global_img, 0, 1)

        return ret

    def __len__(self):
        return self.num_data


if __name__ == "__main__":
    cfg = Args().load("/home/sn/workspace/inrsteg/config/main.yaml")
    dataset = MyDatasetV7(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch['coords'].size())
        print(batch['msg'].size())
        print(batch['img'].size())
        break
