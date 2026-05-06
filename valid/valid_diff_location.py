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
target_sizes = [2048]

# 裁剪区域大小
crop_size = 1024  # 可以修改这个值来改变裁剪区域大小

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

def crop_from_position(img, position, crop_size=128):
    """从指定位置裁剪指定大小的区域"""
    _, _, h, w = img.shape
    
    # 如果图像太小，先resize到crop_size
    if h < crop_size or w < crop_size:
        img = F.interpolate(img, size=(crop_size, crop_size), mode="bilinear")
        return img
    
    if position == "top_left":
        h_start, w_start = 0, 0
    elif position == "top_right":
        h_start, w_start = 0, w - crop_size
    elif position == "bottom_left":
        h_start, w_start = h - crop_size, 0
    elif position == "bottom_right":
        h_start, w_start = h - crop_size, w - crop_size
    elif position == "center":
        h_start = (h - crop_size) // 2
        w_start = (w - crop_size) // 2
    else:
        raise ValueError(f"Unknown position: {position}")
    
    cropped = img[:, :, h_start:h_start+crop_size, w_start:w_start+crop_size]
    return cropped

def process_single_image(img_path, gt_msg_str):
    """处理单张图像"""
    img = read_img(img_path)
    img = transforms.ToTensor()(img).unsqueeze(0).to(device)
    gt_msg = string_to_binary_tensor(gt_msg_str, msg_len).unsqueeze(0).to(device)
    
    results = []
    positions = ["top_left", "top_right", "bottom_left", "bottom_right", "center"]
    
    for target_size in target_sizes:
        # 应用随机失真
        with torch.no_grad():
            noised_img, _ = noiser(img, img)

        # 缩放到目标尺寸
        resized_img = F.interpolate(noised_img, size=(target_size, target_size), mode="bilinear")
        
        # 从五个位置裁剪
        for position in positions:
            cropped_img = crop_from_position(resized_img, position, crop_size)
            
            # 解码
            with torch.no_grad():
                predict_msg = model.decode_msg(cropped_img)
            
            predict_msg = predict_msg[:, :msg_len]
            acc = msg_acc(predict_msg, gt_msg)
            
            results.append({
                'target_size': target_size,
                'position': position,
                'crop_size': crop_size,
                'accuracy': acc.item(),
                'gt_msg': gt_msg_str,
                'pred_msg': ''.join(map(lambda x: str(int(x)), predict_msg.squeeze(0).cpu()))
            })
            
            # 释放显存
            del cropped_img, predict_msg
            torch.cuda.empty_cache()
        
        # 释放显存
        del resized_img, noised_img, _
        torch.cuda.empty_cache()
    
    # 释放显存
    del img, gt_msg
    torch.cuda.empty_cache()
    
    return results

def main():
    img_files = [f for f in os.listdir(input_path) if f.endswith('.png')]
    print(f"Found {len(img_files)} images to process")
    print(f"Crop size: {crop_size}x{crop_size}")
    
    all_results = []
    
    for img_file in tqdm(img_files, desc="Processing images"):
        img_path = os.path.join(input_path, img_file)
        gt_msg_str = img_file.replace('.png', '')
        results = process_single_image(img_path, gt_msg_str)
        all_results.extend(results)
        
        # 每处理几张图像后清理一次显存
        if len(all_results) % 25 == 0:  # 5个位置 * 5张图像
            torch.cuda.empty_cache()
    
    # 统计结果
    print("\n=== Results Summary ===")
    
    # 按尺寸统计
    for target_size in target_sizes:
        size_results = [r for r in all_results if r['target_size'] == target_size]
        avg_acc = sum(r['accuracy'] for r in size_results) / len(size_results)
        print(f"Target size {target_size}x{target_size}: Avg accuracy = {avg_acc:.4f}")
    
    # 按位置统计
    positions = ["top_left", "top_right", "bottom_left", "bottom_right", "center"]
    for position in positions:
        pos_results = [r for r in all_results if r['position'] == position]
        avg_acc = sum(r['accuracy'] for r in pos_results) / len(pos_results)
        print(f"Position {position}: Avg accuracy = {avg_acc:.4f}")
    
    # 按尺寸和位置组合统计
    print("\n=== Detailed Results by Size and Position ===")
    for target_size in target_sizes:
        print(f"\nTarget size {target_size}x{target_size}:")
        for position in positions:
            results = [r for r in all_results if r['target_size'] == target_size and r['position'] == position]
            if results:
                avg_acc = sum(r['accuracy'] for r in results) / len(results)
                print(f"  {position}: {avg_acc:.4f}")
    
    total_avg_acc = sum(r['accuracy'] for r in all_results) / len(all_results)
    print(f"\nOverall average accuracy: {total_avg_acc:.4f}")
    
    # 保存结果
    import json
    with open('decoding_results_by_position.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Detailed results saved to decoding_results_by_position.json")

if __name__ == "__main__":
    main()