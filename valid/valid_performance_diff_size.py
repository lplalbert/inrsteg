from tqdm import tqdm
from FastTools.dataset.dataset import read_img
from FastTools.metre import PSNR
from FastTools.metric.ssim import SSIM
from FastTools.steganography.Noiser.Noiser import Noiser
from FastTools.steganography.utils.common import gen_random_msg, msg_acc
from FastTools.util.ImgUtil import clip_psnr
from FastTools.util.TrainUtil import Args
from dataset.Mydataset import MyDataset, generate_grid_coordinates
from model.ismark_v2 import INRMark
import os
from torchvision import transforms, utils
import torch

# 在不同尺度上进行渲染

cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/main.yaml"
ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/ismark_v1_valid_128_alpha_0.02_min_scale_0.06_pnsr_36/lightning_logs/version_0/checkpoints/ckpt-epoch=79-val_loss=0.3051.ckpt"

device = "cpu"
args = Args().load(cfg_path)
model = INRMark.load_from_checkpoint(ckpt_path, args=args).to(device).eval()

data_path = "/home/light_sun/workspace/inrsteg/data/DIV2K_valid"
imgs = os.listdir(data_path)
fixed_psnr = 36
img_size = 2048
ts = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size))
])


coords = generate_grid_coordinates((-1, -1), 2, img_size).unsqueeze(0).to(device)
total_acc = 0
total_psnr = 0
total_ssim = 0
n = 0
patch_size = 128
overlap = 64

@torch.no_grad()
def render_high_res_image(model, img, coords, msg, patch_size=128, overlap=64, fixed_psnr=None):
    """逐块渲染高分辨率图像"""
    _, _, h, w = img.shape

    output = torch.zeros_like(img)
    count_map = torch.zeros_like(img)  # 用于记录每个像素被渲染的次数

    # 计算块的数量
    num_patches_h = (h - overlap) // (patch_size - overlap)
    num_patches_w = (w - overlap) // (patch_size - overlap)

    # 逐块渲染
    for i in tqdm(range(num_patches_h), desc="Rendering rows"):
        for j in range(num_patches_w):
            # 计算当前块的坐标范围
            h_start = i * (patch_size - overlap)
            w_start = j * (patch_size - overlap)
            h_end = h_start + patch_size
            w_end = w_start + patch_size

            # 提取当前块
            patch = img[:, :, h_start:h_end, w_start:w_end]
            patch_coords = coords[:, h_start:h_end, w_start:w_end, :]
            # print(patch_coords.shape, patch.shape)
            # 渲染当前块
            wm_patch, _ = model.render_img(patch_coords, msg, patch)
            if fixed_psnr:
                wm_patch = clip_psnr(wm_patch, patch, fixed_psnr, over_clip=True)

            # 将渲染结果写入输出图像
            output[:, :, h_start:h_end, w_start:w_end] += wm_patch
            count_map[:, :, h_start:h_end, w_start:w_end] += 1

    # 平均重叠区域
    output = output / count_map
    return output

noiser = Noiser(
    [
        # ("Identity", None),
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
        # ("Dropout", {"prob": [0.3, 0.3]}),
        # ("Color", {"brightness": [0.5, 0.5], "saturation": [0, 0], "hue": [0, 0], "combine":True})
        # ("Rotate", {"angle": [30, 30]})
        # ("Translate", {"translation": [0.1, 0.1]}),
        # ("Scale", {"scale_factor": [1.2, 1.2]})
        # ("GaussianFilter", {"sigma": [2, 2]})
        # ("Rotate", None),
    ]
)
for img in tqdm(imgs):
    img_path = os.path.join(data_path, img)
    img = read_img(img_path)
    img = ts(img).unsqueeze(0).to(device)
    msg = gen_random_msg(args.msg_len).unsqueeze(0).to(device)
    # print(coords.size(), msg.size(), img.size())
    wm_img = render_high_res_image(model, img, coords, msg, patch_size, overlap)
    # wm_img, mask = model.render_img(coords, msg, img)
    # 存储到本地
    # msg_str = ''.join(map(lambda x: str(int(x)), msg))
    # utils.save_image(wm_img, os.path.join("output_imgs", "{},png".format(msg_str)))
    break

    if fixed_psnr:
        wm_img = clip_psnr(wm_img, img, fixed_psnr, over_clip=True)
    noised_img, _ = noiser(wm_img, img)
    predict_msg = model.decode_msg(noised_img)
    acc = msg_acc(predict_msg, msg)
    psnr = PSNR(wm_img, img)
    ssim = SSIM(wm_img, img)
    total_acc += acc.item()
    total_psnr += psnr.item()
    total_ssim += ssim.item()
    n += 1
    pass
print("acc: ", total_acc / n)
print("psnr: ", total_psnr / n)
print("ssim: ", total_ssim / n)
