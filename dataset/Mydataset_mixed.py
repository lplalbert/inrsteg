"""
混合数据集：支持从多个数据目录按权重采样。
用于微调场景，例如 80% document_ds + 20% DIV2K。
"""

import os
from random import choice, random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from FastTools.dataset.dataset import LDataset, read_img
from FastTools.steganography.utils.common import gen_random_msg

from dataset.Mydataset import IMG_EXTENSIONS, gen_start_coords, generate_grid_coordinates


class MixedDataset(LDataset):
    """从多个数据目录按权重采样的混合数据集。

    config 中需新增:
        train_data_paths: ["/path/to/dataset_A", "/path/to/dataset_B"]
        train_data_weights: [0.8, 0.2]
    """

    def __init__(self, args, data_len=1000000, valid=False) -> None:
        super().__init__(args)
        self.use_bg = args.use_bg
        self.num_data = data_len
        self.img_size = args.img_size
        self.msg_len = args.msg_len
        self.use_global = args.use_global
        self.use_cell = args.use_cell
        self.min_scale = args.min_scale

        # 判断是否使用混合模式
        if valid:
            # 验证集只用单个路径
            data_path = args.valid_data_path or "/home/light_sun/workspace/inrsteg/data/DIV2K_valid"
            self.datasets = [(1.0, data_path)]
        else:
            train_paths = getattr(args, "train_data_paths", None)
            train_weights = getattr(args, "train_data_weights", None)
            if train_paths and len(train_paths) > 0:
                if train_weights and len(train_weights) == len(train_paths):
                    weights = [float(w) for w in train_weights]
                else:
                    weights = [1.0] * len(train_paths)
                total = sum(weights)
                self.datasets = [(w / total, p) for w, p in zip(weights, train_paths)]
            else:
                # 回退到单路径模式
                data_path = args.train_data_path or "/home/light_sun/workspace/inrsteg/data/DIV2K_train"
                self.datasets = [(1.0, data_path)]

        # 加载每个数据集的文件列表
        self.file_pools = []
        for weight, path in self.datasets:
            files = sorted([
                name for name in os.listdir(path)
                if name.lower().endswith(IMG_EXTENSIONS)
            ])
            if not files:
                raise FileNotFoundError(f"No image files found in dataset path: {path}")
            self.file_pools.append((weight, path, files))

        self.cache = args.cache
        self.global_ts = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop((self.img_size, self.img_size), scale=(0.06, 1))
        ])

        # 缓存模式
        self.memory_pools = []
        if args.cache:
            for weight, path, files in self.file_pools:
                pool = [read_img(os.path.join(path, name)) for name in files]
                self.memory_pools.append((weight, pool))
        else:
            self.memory_pools = None

    def _sample_image(self):
        """按权重从数据池中采样一张图片。"""
        r = random()
        cumsum = 0.0
        for i, (weight, path, files) in enumerate(self.file_pools):
            cumsum += weight
            if r < cumsum:
                if self.memory_pools:
                    return choice(self.memory_pools[i][1])
                else:
                    return read_img(os.path.join(path, choice(files)))
        # fallback: 最后一个池
        if self.memory_pools:
            return choice(self.memory_pools[-1][1])
        else:
            _, path, files = self.file_pools[-1]
            return read_img(os.path.join(path, choice(files)))

    def __getitem__(self, index):
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
            img = self._sample_image()
            global_img = self.global_ts(img)
            img = global_img
            ret['img'] = torch.clamp(img, 0, 1)
            if self.use_global:
                global_img = F.interpolate(
                    global_img.unsqueeze(0), size=(self.img_size, self.img_size),
                    mode='bilinear', align_corners=True
                ).squeeze(0)
                ret['global_img'] = torch.clamp(global_img, 0, 1)

        return ret

    def __len__(self):
        return self.num_data
