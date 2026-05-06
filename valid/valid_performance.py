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
# cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/main.yaml"
# ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/ismark_v1_valid_128_alpha_0.02_min_scale_1/lightning_logs/version_25/checkpoints/best_1_ckpt-epoch=504-val_loss=0.0355.ckpt"

# cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/v6.yaml"
# ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/ismark_v6/lightning_logs/version_9/checkpoints/ckpt-epoch=24-val_loss=0.0526.ckpt"


cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/v6_30bits.yaml"
ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/ismark_v6_30bits/lightning_logs/version_0/checkpoints/ckpt-epoch=04-val_loss=0.2220.ckpt"


device = "cuda:0"
args = Args().load(cfg_path)
model = INRMark.load_from_checkpoint(ckpt_path, args=args).to(device).eval()

data_path = "/home/light_sun/workspace/inrsteg/data/DIV2K_valid"
imgs = os.listdir(data_path)
fixed_psnr = None
img_size = 128
ts = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_size, img_size))
])
coords = generate_grid_coordinates((-1, -1), 2, img_size).unsqueeze(0).to(device)
total_acc = 0
total_psnr = 0
total_ssim = 0
n = 0
msg_len = 30
noiser = Noiser(
    [
        ("Identity", None),
        # ("GaussianNoise", {"std": 0.05}),
        # ("KorniaJpeg", {"min_q": 50, "max_q": 51}),
        # ("Crop", None),
        # ("Rotate", None),
    ]
)
for img in tqdm(imgs):
    img_path = os.path.join(data_path, img)
    img = read_img(img_path)
    img = ts(img).unsqueeze(0).to(device)
    msg = gen_random_msg(args.msg_len).unsqueeze(0).to(device)
    # print(coords.size(), msg.size(), img.size())
    wm_img, mask = model.render_img(coords, msg, img)
    if fixed_psnr:
        wm_img = clip_psnr(wm_img, img, fixed_psnr, over_clip=True)
    noised_img, _ = noiser(wm_img, img)
    predict_msg = model.decode_msg(noised_img)
    # 只要前30位
    predict_msg = predict_msg[:, :msg_len]
    msg = msg[:, :msg_len]
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
