from tqdm import tqdm
import sys
sys.path.append("/home/light_sun/workspace/inrmark_2/inrsteg-final_v1")
from FastTools.dataset.dataset import read_img
from FastTools.metre import PSNR
from FastTools.metric.ssim import SSIM
from FastTools.steganography.Noiser.Noiser import Noiser
from FastTools.steganography.utils.common import gen_random_msg, msg_acc
from FastTools.util.ImgUtil import clip_psnr
from FastTools.util.TrainUtil import Args
from dataset.Mydataset import MyDataset, generate_grid_coordinates
from model.ismark_v6_30bit import INRMark
import os
from torchvision import transforms, utils
import torch
import torch.nn.functional as F
import random

# 配置
cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/v6_30bits.yaml"
ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/ismark_v6_30bits/lightning_logs/version_0/checkpoints/ckpt-epoch=109-val_loss=0.0481.ckpt"
device = "cuda:1"
args = Args().load(cfg_path)
model = INRMark.load_from_checkpoint(ckpt_path, args=args).to(device).eval()

# 输入输出路径
input_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/div2k_imgs"

msg_len = 30

# 不同尺寸的缩放目标
target_sizes = [128, 512, 2048, 4096]


# 测量不同社交媒体上的图片
input_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/print_image"
target_sizes = [128]

# 失真配置
noiser = Noiser(
    [
        ("Identity", None),
        # ("GaussianNoise", {"std": 0.01}),
        # ("GaussianNoise", {"std": 0.05}),
        # ("KorniaJpeg", {"min_q": 50, "max_q": 51}),
        # ("KorniaJpeg", {"min_q": 60, "max_q": 61}),
        # ("KorniaJpeg", {"min_q": 80, "max_q": 81}),
        # ("Crop", {"ratio": [0.6, 0.6]}),
        # ("Crop", {"ratio": [0.8, 0.8]}),
        # ("Cropout", {"ratio": [0.1, 0.1]}),
        # ("Cropout", {"ratio": [0.05, 0.05]}),
        # ("Dropout", {"prob": [0.1, 0.1]}),
        # ("Dropout", {"prob": [0.5, 0.5]}),
        # ("Color", {"brightness": [0.1, 0.1], "saturation": [0.1, 0.1], "hue": [0.1, 0.1], "combine":True}),
        # ("Rotate", {"angle": [10, 10]}),
        # ("Translate", {"translation": [0.1, 0.1]}),
        # ("Scale", {"scale_factor": [1.2, 1.2]}),
        # ("GaussianFilter", {"sigma": [2, 2]}),
        # ("Rotate", None),
    ]
)

def string_to_binary_tensor(msg_str, msg_len=30):
    """将字符串转换为二进制张量"""
    if len(msg_str) < msg_len:
        msg_str = msg_str.ljust(msg_len, '0')
    elif len(msg_str) > msg_len:
        msg_str = msg_str[:msg_len]
    
    binary_list = []
    for char in msg_str:
        binary_list.append(int(char))
    
    return torch.tensor(binary_list, dtype=torch.float32)

def random_crop_128x128(img):
    """随机裁剪128x128的区域"""
    _, _, h, w = img.shape
    
    if h < 128 or w < 128:
        img = F.interpolate(img, size=(128, 128), mode="bilinear")
        return img
    
    h_start = random.randint(0, h - 128)
    w_start = random.randint(0, w - 128)
    cropped = img[:, :, h_start:h_start+128, w_start:w_start+128]
    return cropped

def process_single_image(img_path, gt_msg_str):
    """处理单张图像"""
    img = read_img(img_path)
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    gt_msg = string_to_binary_tensor(gt_msg_str, msg_len).unsqueeze(0).to(device)
    
    results = []
    
    for target_size in target_sizes:
        # 应用随机失真
        with torch.no_grad():
            noised_img, _ = noiser(img, img)

        # 缩放到目标尺寸
        resized_img = F.interpolate(noised_img, size=(target_size, target_size), mode="bilinear")
        
        # 随机裁剪一次128x128区域
        cropped_img = random_crop_128x128(resized_img)
        
        # 解码
        with torch.no_grad():
            predict_msg = model.decode_msg(cropped_img)
        
        predict_msg = predict_msg[:, :msg_len]
        acc = msg_acc(predict_msg, gt_msg)
        # 释放显存
        del resized_img, noised_img, cropped_img, _
        torch.cuda.empty_cache()

        results.append({
            'target_size': target_size,
            'noise_type': "None",
            'accuracy': acc.item(),
            'gt_msg': gt_msg_str,
            'pred_msg': ''.join(map(lambda x: str(int(x)), predict_msg.squeeze(0).cpu()))
        })
    
    return results

def main():
    img_files = [f for f in os.listdir(input_path) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
    print(f"Found {len(img_files)} images to process")
    
    all_results = []
    
    for img_file in tqdm(img_files, desc="Processing images"):
        img_path = os.path.join(input_path, img_file)
        gt_msg_str = img_file.replace('.png', '')
        results = process_single_image(img_path, gt_msg_str)
        all_results.extend(results)
    
    # 统计结果
    print("\n=== Results Summary ===")
    
    for target_size in target_sizes:
        size_results = [r for r in all_results if r['target_size'] == target_size]
        avg_acc = sum(r['accuracy'] for r in size_results) / len(size_results)
        print(f"Target size {target_size}x{target_size}: Avg accuracy = {avg_acc:.4f}")
    
    # 修复：将noise_type转换为字符串进行比较
    noise_types = list(set(str(r['noise_type']) for r in all_results))
    for noise_type in noise_types:
        noise_results = [r for r in all_results if str(r['noise_type']) == noise_type]
        avg_acc = sum(r['accuracy'] for r in noise_results) / len(noise_results)
        print(f"Noise type {noise_type}: Avg accuracy = {avg_acc:.4f}")
    
    total_avg_acc = sum(r['accuracy'] for r in all_results) / len(all_results)
    print(f"\nOverall average accuracy: {total_avg_acc:.4f}")
    

if __name__ == "__main__":
    main()