import sys
sys.path.append("/home/light_sun/workspace/inrmark_2/inrsteg-final_v1")

import torch
import torch.nn.functional as F
from torchvision import transforms
import os
from tqdm import tqdm
from PIL import Image

import torchattacks

from FastTools.dataset.dataset import read_img
from FastTools.steganography.utils.common import msg_acc
from FastTools.util.TrainUtil import Args
from model.ismark_v6_30bit import INRMark
import math

# 配置
cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/v6_30bits.yaml"
ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/ismark_v6_30bits/lightning_logs/version_0/checkpoints/ckpt-epoch=109-val_loss=0.0481.ckpt"
device = "cuda:3"
args = Args().load(cfg_path)
model = INRMark.load_from_checkpoint(ckpt_path, args=args).to(device).eval()

input_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/div2k_imgs"
output_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/adversarial_samples"
os.makedirs(output_path, exist_ok=True)

msg_len = 30

def string_to_binary_tensor(msg_str, msg_len=30):
    if len(msg_str) < msg_len:
        msg_str = msg_str.ljust(msg_len, '0')
    elif len(msg_str) > msg_len:
        msg_str = msg_str[:msg_len]
    binary_list = [int(char) for char in msg_str]
    return torch.tensor(binary_list, dtype=torch.float32)


# 解码器适配
class DecoderWrapper(torch.nn.Module):
    def __init__(self, model, msg_len=30):
        super().__init__()
        self.model = model
        self.msg_len = msg_len

    def forward(self, x):
        # x: [B, C, H, W]
        if x.shape[2] != 128 or x.shape[3] != 128:
            x = F.interpolate(x, size=(128, 128), mode="bilinear")
        out = self.model.decode_msg(x)[:, :self.msg_len]
        # 这里我们将二进制向量转为概率，适配攻击接口
        return out

def calc_psnr(img1, img2):
    # img1, img2: [1, C, H, W] or [C, H, W], 取值范围[0,1]
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))


attack_configs = {
    'pgd_weak': {'type': 'PGD', 'eps': 8/255, 'alpha': 2/255, 'steps': 20},  # 30PNSR，99.9，99.4
    'pgd_medium': {'type': 'PGD', 'eps': 16/255, 'alpha': 4/255, 'steps': 20},  # 26.91PNSR 99.9 82.07
    'fgsm_weak': {'type': 'FGSM', 'eps': 16/255},  # 38.39PNSR 99.9 99.9
    'fgsm_strong': {'type': 'FGSM', 'eps': 128/255},  # 22.82PNSR 99.9 99.9
    'bim': {'type': 'BIM', 'eps': 64/255, 'alpha': 8/255, 'steps': 20},  # 28.05PNSR 99.9 99.9
    'bim_medium': {'type': 'BIM', 'eps': 128/255, 'alpha': 16/255, 'steps': 20},  # 28.05PNSR 99.9 99.9

}

def main():
    wrapped_model = DecoderWrapper(model, msg_len).to(device)
    wrapped_model.eval()

    # # 选择攻击方法
    # atk = torchattacks.PGD(wrapped_model, eps=8/255, alpha=2/255, steps=20) # 30PNSR，99.9，99.4
    # # atk = torchattacks.PGD(wrapped_model, eps=0.08, alpha=0.005, steps=20) # 26.91PNSR 99.9 82.07
    # # atk = torchattacks.PGD(wrapped_model, eps=0.1, alpha=0.01, steps=20) # 25.01PNSR 99.9 61.6
    # # atk = torchattacks.FGSM(wrapped_model, eps=0.1) # 38.39PNSR 99.9 99.9
    # # atk = torchattacks.FGSM(wrapped_model, eps=0.5) # 25.88PNSR 99.9 99.9
    # # atk = torchattacks.FGSM(wrapped_model, eps=2) # 22.82PNSR 99.9 99.9
    # # atk = torchattacks.BIM(wrapped_model, eps=1, alpha=0.1, steps=20) # 28.05PNSR 99.9 99.9




    img_files = [f for f in os.listdir(input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(img_files)} images to process")

    for name, config in attack_configs.items():
        if config['type'] == 'PGD':
            atk = torchattacks.PGD(wrapped_model, eps=config['eps'], alpha=config['alpha'], steps=config['steps'])
        elif config['type'] == 'FGSM':
            atk = torchattacks.FGSM(wrapped_model, eps=config['eps'])
        elif config['type'] == 'BIM':
            atk = torchattacks.BIM(wrapped_model, eps=config['eps'], alpha=config['alpha'], steps=config['steps'])
        
        total_psnr = 0.0
        total_acc = 0.0
        total_adv_acc = 0.0
        img_files = img_files
        num_imgs = len(img_files)

        for img_file in tqdm(img_files):
            img_path = os.path.join(input_path, img_file)
            gt_msg_str = img_file.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')

            img = read_img(img_path)
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
            gt_msg = string_to_binary_tensor(gt_msg_str, msg_len).unsqueeze(0).to(device)

            # 原始准确率
            with torch.no_grad():
                pred = model.decode_msg(img_tensor)[:, :msg_len]
                acc = msg_acc(pred, gt_msg).item()
            # print(f"{img_file} 原始准确率: {acc:.4f}")
            total_acc += acc

            # 生成对抗样本
            adv_img = atk(img_tensor, gt_msg)
            adv_img = torch.clamp(adv_img, 0, 1)

            # 计算PSNR
            psnr = calc_psnr(img_tensor, adv_img)
            # print(f"{img_file} 攻击前后PSNR: {psnr:.2f} dB")
            total_psnr += psnr

            # 对抗样本准确率
            with torch.no_grad():
                adv_pred = model.decode_msg(adv_img)[:, :msg_len]
                adv_acc = msg_acc(adv_pred, gt_msg).item()
            # print(f"{img_file} 对抗后准确率: {adv_acc:.4f}")
            total_adv_acc += adv_acc

            # # 保存对抗样本
            # adv_img_pil = transforms.ToPILImage()(adv_img.squeeze(0).cpu())
            # adv_img_pil.save(os.path.join(output_path, img_file.replace('.png', '_adv.png')))

        # 统计并打印平均值
        print(f"攻击方法: {name}")
        print(f"平均PSNR: {total_psnr / num_imgs:.2f} dB")
        print(f"平均攻击前准确率: {total_acc / num_imgs:.4f}")
        print(f"平均攻击后准确率: {total_adv_acc / num_imgs:.4f}")

if __name__ == "__main__":
    main()