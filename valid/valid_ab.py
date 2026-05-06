import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm

from FastTools.metre import PSNR
from FastTools.metric.ssim import SSIM
from FastTools.steganography.Noiser.Noiser import Noiser
from FastTools.steganography.utils.common import msg_acc
from FastTools.util.TrainUtil import Args
from dataset.Mydataset import MyDataset

# 早融合
# from model.ismark_v6_remove_fft_early_fusion import INRMark
# cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/v6_early_fusion.yaml"
# ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/v6_early_fusion/lightning_logs/version_0/checkpoints/ckpt-epoch=634-val_loss=0.0092.ckpt"


#只有坐标变换的
# from model.ismark_v6_remove_fft_early_fusion import INRMark
# cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/v6_early_fusion.yaml"
# ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/v6_early_fusion/lightning_logs/version_0/checkpoints/ckpt-epoch=634-val_loss=0.0092.ckpt"

# from model.ismark_v6_50bit import INRMark
# cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/v6_50bits.yaml"
# ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/ismark_v6_50bits/lightning_logs/version_0/checkpoints/ckpt-epoch=19-val_loss=0.4347.ckpt"

# from model.ismark_v6_30bit import INRMark
# cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/v6_30bits.yaml"
# ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/ismark_v6_30bits/lightning_logs/version_0/checkpoints/ckpt-epoch=109-val_loss=0.0481.ckpt"

from model.ismark_v6_100bit import INRMark
cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/v6_100bits.yaml"
ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/ismark_v6_100bits_fixed_psnr/lightning_logs/version_0/checkpoints/ckpt-epoch=29-val_loss=0.6403.ckpt"


device = "cuda:1"
args = Args().load(cfg_path)
dataset = MyDataset(args, 200, True)

model = INRMark.load_from_checkpoint(ckpt_path, args=args).to(device).eval()
msg_len = 100
model.fixed_psnr =31
noiser = Noiser([
            ("Identity", None),
            ("Rotate", None),
            ("Crop", None),
            ("Translate", None),
            ("Scale", None),
            ("Shear", None),
            ("Dropout", None),
            ("Cropout", None),

            ("Color", None),
            ("KorniaJpeg", None),
            ("GaussianFilter", None),
            ("GaussianNoise", None),

        ])

total_acc = 0
total_psnr = 0
total_ssim = 0
n = 0
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
for batch in tqdm(dataloader):
    coords = batch['coords'].to(device)
    msg = batch['msg'].to(device)
    cover_img = batch['img'].to(device)

    res = model(coords, msg, cover_img, over_clip=True)

    wm_img = res['wm_img']
    predict_msg = res['predict_msg']
    noised_img = res['noised_img']    
    msg = msg[:, :msg_len]
    predict_msg = predict_msg[:,:msg_len]
    acc = msg_acc(predict_msg, msg)
    psnr = PSNR(wm_img, cover_img)
    ssim = SSIM(wm_img, cover_img)
    total_acc += acc.item()
    total_psnr += psnr.item()
    total_ssim += ssim.item()
    n += 1
    pass

print("acc: ", total_acc / n)
print("psnr: ", total_psnr / n)
print("ssim: ", total_ssim / n)


