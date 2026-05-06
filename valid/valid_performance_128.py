from tqdm import tqdm
from FastTools.dataset.dataset import read_img
from FastTools.metre import PSNR
from FastTools.metric.ssim import SSIM
from FastTools.steganography.Noiser.Noiser import Noiser
from FastTools.steganography.utils.common import gen_random_msg, msg_acc
from FastTools.util.ImgUtil import clip_psnr
from FastTools.util.TrainUtil import Args
from dataset.Mydataset import MyDataset, generate_grid_coordinates
from model.ismark_v1 import INRMark
import os
from torchvision import transforms, utils

# 仅在128尺度上进行验证

cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/main.yaml"
ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/ismark_v1_valid_128_alpha_0.02_min_scale_1/lightning_logs/version_25/checkpoints/ckpt-epoch=504-val_loss=0.0355.ckpt"

device = "cuda:0"
args = Args().load(cfg_path)
model = INRMark.load_from_checkpoint(ckpt_path, args=args).to(device).eval()

data_path = "/home/light_sun/workspace/inrsteg/data/DIV2K_valid"
imgs = os.listdir(data_path)
fixed_psnr = None

ts = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])
coords = generate_grid_coordinates((-1, -1), 2, 128).unsqueeze(0).to(device)
total_acc = 0
total_psnr = 0
total_ssim = 0
n = 0
noiser = Noiser(
    [
        # ("Identity", None),
        # ("GaussianNoise", {"std": 0.01}),
        # ("GaussianNoise", {"std": 0.05}),
        ("KorniaJpeg", {"min_q": 50, "max_q": 51}),
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
    wm_img, mask = model.render_img(coords, msg, img)
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
