import sys
sys.path.append("/home/light_sun/workspace/inrmark_2/inrsteg-final_v1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import psutil
import GPUtil
from model.ismark_v6 import INRMark
from FastTools.util.TrainUtil import Args

class LightweightCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

def test_gpu_memory_2k():
    """测试2K图像在3层卷积上的GPU内存消耗"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建2K图像 (2048x2048)
    batch_size = 2
    # img_size = 2048
    # x = torch.randn(batch_size, 3, img_size, img_size, requires_grad=True).to(device)
    
    # 创建模型
    args = Args().load("/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/main.yaml")

    model = INRMark(args).to(device)
    img_size = 128
    msg_len = 30
    
    # model = LightweightCNN().to(device)
    
    # 记录初始GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        print(f"Initial GPU memory: {initial_memory:.3f} GB")
    
    # 前向传播
    print("Starting forward pass...")
    start_time = time.time()
    
    
    res = model(
        torch.clamp(torch.rand(batch_size, img_size, img_size, 2, requires_grad=True), -1, 1).to(device),
        torch.rand(batch_size, msg_len, requires_grad=True).to(device),
        torch.clamp(torch.randn(batch_size, 3, img_size, img_size, requires_grad=True), 0, 1).to(device))

    # output = model(x)
    
    forward_time = time.time() - start_time
    
    # 记录前向传播后的GPU内存
    if torch.cuda.is_available():
        forward_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        print(f"GPU memory after forward pass: {forward_memory:.3f} GB")
        print(f"Memory used by forward pass: {forward_memory - initial_memory:.3f} GB")
    
    # 反向传播测试
    print("Starting backward pass...")
    start_time = time.time()

    wm_img = res['wm_img']
    predict_msg = res['predict_msg']
    noised_img = res['noised_img']
    cover_img = torch.clamp(torch.randn(batch_size, 3, img_size, img_size), 0, 1).to(device)
    msg = torch.rand(batch_size, msg_len).to(device)
    msg_loss = F.mse_loss(predict_msg, msg)

    img_loss = F.mse_loss(wm_img, cover_img)
    # lpips_loss = self.w_lpips * torch.mean(self.lpips.forward(wm_img*2-1,cover_img*2-1))

    loss = msg_loss + img_loss  # + lpips_loss
    loss.backward()
    
    backward_time = time.time() - start_time
    
    # 记录反向传播后的GPU内存
    if torch.cuda.is_available():
        backward_memory = torch.cuda.memory_allocated() / 1024  # GB
        print(f"GPU memory after backward pass: {backward_memory:.3f} GB")
        print(f"Total memory used: {backward_memory - initial_memory:.3f} GB")
    
    # 获取GPU信息
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        print(f"\nGPU: {gpu.name}")
        print(f"GPU memory total: {gpu.memoryTotal} MB")
        print(f"GPU memory used: {gpu.memoryUsed} MB")
        print(f"GPU memory free: {gpu.memoryFree} MB")

        # 计算每个平均GB
        print(f"Average memory used per batch: {gpu.memoryUsed / batch_size / 1024} GB")
    
    print(f"\nTiming:")
    print(f"Forward pass time: {forward_time:.3f} seconds")
    print(f"Backward pass time: {backward_time:.3f} seconds")
    

if __name__ == "__main__":
    # 安装依赖: pip install GPUtil
    try:
        import GPUtil
    except ImportError:
        print("Please install GPUtil: pip install GPUtil")
        exit(1)
    
    results = test_gpu_memory_2k()
