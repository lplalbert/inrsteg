from tqdm import tqdm
import sys
import time
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

# 在不同尺度上进行渲染

cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/v6_30bits.yaml"
ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/ismark_v6_30bits/lightning_logs/version_0/checkpoints/ckpt-epoch=109-val_loss=0.0481.ckpt"
device = "cuda:3"
args = Args().load(cfg_path)
model = INRMark.load_from_checkpoint(ckpt_path, args=args).to(device).eval()

data_path = "/home/light_sun/workspace/inrsteg/data/DIV2K_valid"
imgs = os.listdir(data_path)
fixed_psnr = None
img_size = 2048
ts = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size))
])

msg_len = 30

coords = generate_grid_coordinates((-1, -1), 2, img_size).unsqueeze(0).to(device)
total_acc = 0
total_psnr = 0
total_ssim = 0
total_inference_time = 0
n = 0
patch_size = 128
overlap = 0

@torch.no_grad()
def render_high_res_image_parallel(model, img, coords, msg, patch_size=128, overlap=64, fixed_psnr=None):
    """并行分块渲染高分辨率图像"""
    _, _, h, w = img.shape

    # 计算块的数量
    num_patches_h = (h - overlap) // (patch_size - overlap)
    num_patches_w = (w - overlap) // (patch_size - overlap)
    
    # 准备所有块的输入
    patches = []
    patch_coords_list = []
    positions = []
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # 计算当前块的坐标范围
            h_start = i * (patch_size - overlap)
            w_start = j * (patch_size - overlap)
            h_end = h_start + patch_size
            w_end = w_start + patch_size

            # 提取当前块
            patch = img[:, :, h_start:h_end, w_start:w_end]
            patch_coords = coords[:, h_start:h_end, w_start:w_end, :]
            
            patches.append(patch)
            patch_coords_list.append(patch_coords)
            positions.append((h_start, h_end, w_start, w_end))
    
    # 批量处理所有块
    if patches:
        patches_batch = torch.cat(patches, dim=0)
        patch_coords_batch = torch.cat(patch_coords_list, dim=0)
        msg_batch = msg.repeat(len(patches), 1)
        
        # 并行渲染所有块
        wm_patches, _ = model.render_img(patch_coords_batch, msg_batch, patches_batch)
        
        if fixed_psnr:
            wm_patches = clip_psnr(wm_patches, patches_batch, fixed_psnr, over_clip=True)
        
        # 重构输出图像
        output = torch.zeros_like(img)
        count_map = torch.zeros_like(img)
        
        for idx, (h_start, h_end, w_start, w_end) in enumerate(positions):
            wm_patch = wm_patches[idx:idx+1]
            output[:, :, h_start:h_end, w_start:w_end] += wm_patch
            count_map[:, :, h_start:h_end, w_start:w_end] += 1
        
        # 平均重叠区域
        output = output / count_map
        return output
    else:
        return img

noiser = Noiser(
    [
        ("Identity", None),
        ("GaussianNoise", {"std": 0.01}),
        ("GaussianNoise", {"std": 0.05}),
        ("KorniaJpeg", {"min_q": 50, "max_q": 51}),
        ("KorniaJpeg", {"min_q": 60, "max_q": 61}),
        ("KorniaJpeg", {"min_q": 80, "max_q": 81}),
        ("Crop", {"ratio": [0.6, 0.6]}),
        ("Crop", {"ratio": [0.8, 0.8]}),
        ("Cropout", {"ratio": [0.1, 0.1]}),
        ("Cropout", {"ratio": [0.05, 0.05]}),
        ("Dropout", {"prob": [0.1, 0.1]}),
        ("Dropout", {"prob": [0.3, 0.3]}),
        ("Color", {"brightness": [0.5, 0.5], "saturation": [0, 0], "hue": [0, 0], "combine":True}),
        ("Rotate", {"angle": [30, 30]}),
        ("Translate", {"translation": [0.1, 0.1]}),
        ("Scale", {"scale_factor": [1.2, 1.2]}),
        ("GaussianFilter", {"sigma": [2, 2]}),
        ("Rotate", None),
    ]
)

for img in tqdm(imgs):
    img_path = os.path.join(data_path, img)
    img = read_img(img_path)
    img = ts(img).unsqueeze(0).to(device)
    msg = gen_random_msg(args.msg_len).unsqueeze(0).to(device)
    
    # 测量推理时间

    start_time = time.time()
    wm_img = render_high_res_image_parallel(model, img, coords, msg, patch_size, overlap)
    end_time = time.time()
    inference_time_ms = (end_time - start_time) * 1000
    print(f"Inference time: {inference_time_ms} ms")
    if n != 0:
        total_inference_time += inference_time_ms
    
    if fixed_psnr:
        wm_img = clip_psnr(wm_img, img, fixed_psnr, over_clip=True)
    noised_img, _ = noiser(wm_img, img)
    # 缩放到128x128
    noised_img = F.interpolate(noised_img, size=(128, 128), mode="bilinear")
    with torch.no_grad():
        predict_msg = model.decode_msg(noised_img)
    predict_msg = predict_msg[:, :msg_len]
    msg = msg[:, :msg_len]
    acc = msg_acc(predict_msg, msg)
    psnr = PSNR(wm_img, img)
    ssim = SSIM(wm_img, img)
    total_acc += acc.item()
    total_psnr += psnr.item()
    total_ssim += ssim.item()

    # msg_str = ''.join(map(lambda x: str(int(x)), msg.squeeze(0).cpu()))
    # utils.save_image(wm_img, os.path.join("/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/div2k_imgs", "{}.png".format(msg_str)))
    n += 1

print("acc: ", total_acc / n)
print("psnr: ", total_psnr / n)
print("ssim: ", total_ssim / n)
print("Average inference time: {:.2f} ms".format(total_inference_time / (n-1)))