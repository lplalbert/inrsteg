
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

def get_cfg_value(args, key, default=None):
    value = getattr(args, key)
    return default if value is None else value

def find_image_files(data_path):
    files = []
    for root, _, names in os.walk(data_path):
        for name in names:
            if name.lower().endswith(IMG_EXTENSIONS):
                files.append(os.path.join(root, name))
    files.sort()
    return files

def resolve_min_scale(args):
    target_crop_size = get_cfg_value(args, "target_crop_size", None)
    target_canvas_size = get_cfg_value(args, "target_canvas_size", None)
    if target_crop_size and target_canvas_size:
        min_scale = float(target_crop_size) / float(target_canvas_size)
    else:
        min_scale = float(get_cfg_value(args, "min_scale", 0.0))
    return float(np.clip(min_scale, 1e-6, 1.0))

def gen_start_coords(min_scale=0):
    min_scale = float(np.clip(min_scale, 1e-6, 1.0))
    min_side_length = 2 * min_scale
    width = np.random.uniform(min_side_length, 2)
    start_x = np.random.uniform(-1, 1 - width)
    start_y = np.random.uniform(-1, 1 - width)
    return (start_x, start_y), width
    pass

def generate_grid_coordinates(top_left, side_length, grid_size=256):
    """
    生成固定大小的坐标矩阵
    
    :param top_left: 左上角点的坐标 (x1, y1)
    :param side_length: 矩阵的边长
    :param grid_size: 坐标矩阵的大小 (默认 256x256)
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
        self.sources = self._build_sources(args, data_len, valid)
        self.num_data = sum(source["length"] for source in self.sources)
        self.cache = args.cache     
        self.global_ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.06, 1))
        ])
        self.min_scale = resolve_min_scale(args)
        self.memory = None
        if args.cache:
            self.memory = [
                [read_img(path) for path in source["files"]]
                for source in self.sources
            ]
        pass

    def _build_sources(self, args, data_len, valid):
        if valid:
            data_path = args.valid_data_path or "/home/light_sun/workspace/inrsteg/data/DIV2K_valid"
            files = find_image_files(data_path)
            if not files:
                raise FileNotFoundError("No image files found in dataset path: {}".format(data_path))
            return [{"path": data_path, "files": files, "length": data_len}]

        train_data_paths = get_cfg_value(args, "train_data_paths", None)
        if train_data_paths is None:
            data_path = args.train_data_path or "/home/light_sun/workspace/inrsteg/data/DIV2K_train"
            files = find_image_files(data_path)
            if not files:
                raise FileNotFoundError("No image files found in dataset path: {}".format(data_path))
            return [{"path": data_path, "files": files, "length": data_len}]

        train_data_lengths = get_cfg_value(args, "train_data_lengths", None)
        if train_data_lengths is None:
            raise ValueError("train_data_lengths must be set when train_data_paths is used.")
        if len(train_data_paths) != len(train_data_lengths):
            raise ValueError("train_data_paths and train_data_lengths must have the same length.")

        sources = []
        for data_path, source_len in zip(train_data_paths, train_data_lengths):
            files = find_image_files(data_path)
            if not files:
                raise FileNotFoundError("No image files found in dataset path: {}".format(data_path))
            sources.append({
                "path": data_path,
                "files": files,
                "length": int(source_len),
            })
        return sources

    def _select_source(self, index):
        offset = index % self.num_data
        for source_idx, source in enumerate(self.sources):
            if offset < source["length"]:
                return source_idx, source
            offset -= source["length"]
        return len(self.sources) - 1, self.sources[-1]
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
            source_idx, source = self._select_source(index)
            if self.cache:
                img = choice(self.memory[source_idx])
            else:
                img = read_img(choice(source["files"]))
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
