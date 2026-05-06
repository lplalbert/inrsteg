import random
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

# 在不同尺度上进行渲染
cfg_path = "config/v6_30bits.yaml"
ckpt_path = "output/ismark_v6_30bits/lightning_logs/version_0/checkpoints/ckpt-epoch=109-val_loss=0.0481.ckpt"
device = "cpu"
args = Args().load(cfg_path)
model = INRMark.load_from_checkpoint(ckpt_path, args=args).to(device).eval()

data_path = "/mnt/xsj2023/Datasets/DIV2K/DIV2K_valid"
imgs = os.listdir(data_path)
fixed_psnr = None
img_size = 2048
ts = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size))
])

msg_len = 30
padding_len = 100

coords = generate_grid_coordinates((-1, -1), 2, img_size).unsqueeze(0).to(device)
total_acc = 0
total_psnr = 0
total_ssim = 0
n = 0
patch_size = 128
overlap = 0

@torch.no_grad()
def render_high_res_image(model, coords, msg, h, w, patch_size=128, overlap=64, fixed_psnr=None):
    """逐块渲染高分辨率图像"""
    output = torch.zeros((1, 3, h, w)).to(model.device)
    count_map = torch.zeros((1, 3, h, w)).to(model.device)  # 用于记录每个像素被渲染的次数

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
            patch_coords = coords[:, h_start:h_end, w_start:w_end, :]
            # print(patch_coords.shape, patch.shape)
            # 渲染当前块
            wm_patch, _ = model.render_img(patch_coords, msg)
            # 将渲染结果写入输出图像
            output[:, :, h_start:h_end, w_start:w_end] += wm_patch
            count_map[:, :, h_start:h_end, w_start:w_end] += 1

    # 平均重叠区域
    output = output / count_map
    return output






def string_to_binary_tensor(msg_str, msg_len=30, padding_len=100):
    """将字符串转换为二进制张量"""
    if len(msg_str) < msg_len:
        msg_str = msg_str.ljust(msg_len, '0')
    elif len(msg_str) > msg_len:
        msg_str = msg_str[:msg_len]
    
    binary_list = []
    for char in msg_str:
        binary_list.append(int(char))

    msg =  torch.tensor(binary_list, dtype=torch.float32)
    if msg.size(0) < padding_len:
        msg = torch.cat([msg, torch.zeros(padding_len - msg.size(0), dtype=torch.float32)], dim=0)
    elif msg.size(0) > padding_len:
        msg = msg[:padding_len]
    
    return msg



def generate_watermark_template(msg, save_path, device="cuda:0" if torch.cuda.is_available() else "cpu", img_size=2048):
    assert len(msg) == msg_len
    model.to(device)
    msg = string_to_binary_tensor(msg, msg_len, padding_len).unsqueeze(0).to(device)
    coords = generate_grid_coordinates((-1, -1), 2, img_size).unsqueeze(0).to(device)
    wm_img = render_high_res_image(model, coords, msg, img_size, img_size)
    # torch tensor保存
    torch.save(wm_img, save_path)
    return wm_img


def encode_watermark(watermark_path, img_path, save_path=None, device="cuda:0" if torch.cuda.is_available() else "cpu"):

    img = read_img(img_path)
    img = ts(img).unsqueeze(0).to(device)
    watermark_template = torch.load(watermark_path, map_location="cpu").to(device)
    max_size=max(img.size(2), img.size(3))
    watermark=F.interpolate(watermark_template, size=(max_size, max_size), mode="bilinear")
    img_h, img_w = img.size(2), img.size(3)
    
    print(img_h, img_w)
    
    # 适配watermark，长边对齐img，然后中心对称裁剪
    wm_h, wm_w = watermark.size(2), watermark.size(3)
    h_start = (wm_h - img_h) // 2
    w_start = (wm_w - img_w) // 2
    watermark = watermark[:, :, h_start:h_start+img_h, w_start:w_start+img_w]
    
    # 现在watermark和img的尺寸完全一致
    assert watermark.shape == img.shape, f"Watermark shape {watermark.shape} != img shape {img.shape}"
    watermarked= img + watermark
    if save_path is not None:
        utils.save_image(watermarked, save_path)
    return watermarked

    pass


def decode_watermark(img, device="cpu"):
    model.to(device)
    if isinstance(img, str):
        noised_img = read_img(img_path)
        noised_img = ts(noised_img).unsqueeze(0).to(device)
    else:
        noised_img = img.to(device)
    noised_img = F.interpolate(noised_img, size=(128, 128), mode="bilinear")
    with torch.no_grad():
        predict_msg = model.decode_msg(noised_img)
        predict_msg = torch.round(predict_msg).clamp(0, 1)
    return "".join(map(lambda x: str(int(x)), predict_msg.squeeze(0).cpu()))[:msg_len]


if __name__ == "__main__":



    ### 生成模版
    # 随机产生30位二进制字符串
    # msg = gen_random_msg(100)
    
    msg = "".join(random.choices("01", k=30))

    # save_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/watermark_template/{}.pt".format(msg)
    # generate_watermark_template(msg, save_path, device="cuda:2")


    # ### 嵌入水印 / 提取水印

    # img_dir_path = "/home/light_sun/workspace/inrsteg/data/DIV2K_valid"
    # watermark_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/watermark_template/{}.pt".format(msg)
    # watermark_msg = msg
    
    
    template_path = "/home/xsj2023/output_watermark_tmp/{}.pt".format(msg)
    os.makedirs(os.path.dirname(template_path), exist_ok=True)
    generate_watermark_template(msg, template_path, device= "cuda:2")


    ### 嵌入水印 / 提取水印

    img_dir_path = "./output/div2k_imgs"
    watermark_path = template_path
    watermark_msg = msg
    watermarked_dir = "/home/xsj2023/output_watermarked_img_tmp"
    os.makedirs(watermarked_dir, exist_ok=True)
    
    
    
    

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

    for img_name in os.listdir(img_dir_path):
        img_path = os.path.join(img_dir_path, img_name)
        stem = os.path.splitext(img_name)[0]
        watermarked_path = os.path.join(watermarked_dir, "{}_{}.png".format(stem, msg))

        encode_watermark(watermark_path, img_path, save_path=watermarked_path, device="cuda:2")

        watermarked = read_img(watermarked_path)
        watermarked = ts(watermarked).unsqueeze(0).to(device)
        
        # 随机裁剪一个128的块, 模拟裁剪
        watermarked = random_crop_128x128(watermarked)
        
        predict_msg = decode_watermark(watermarked, device="cuda:2")
        
        
        print(predict_msg)


        # 计算acc
        acc = msg_acc(string_to_binary_tensor(predict_msg, msg_len, msg_len), string_to_binary_tensor(watermark_msg, msg_len, msg_len))
        print(acc)

    pass
