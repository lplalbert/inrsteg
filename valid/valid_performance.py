from tqdm import tqdm
from FastTools.dataset.dataset import read_img
from FastTools.metre import PSNR
from FastTools.metric.ssim import SSIM
from FastTools.steganography.utils.common import gen_random_msg, msg_acc
from FastTools.util.ImgUtil import clip_psnr
from FastTools.util.TrainUtil import Args
from dataset.Mydataset import MyDataset, generate_grid_coordinates
from model.ismark_v1 import INRMark
import os
from torchvision import transforms, utils
cfg_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/config/main.yaml"
ckpt_path = "/home/light_sun/workspace/inrmark_2/inrsteg-final_v1/output/ismark_v1_fix_psnr_36_alpha_0.04/lightning_logs/version_0/checkpoints/ckpt-epoch=19-val_loss=0.0123.ckpt"

device = "cuda:0"
args = Args().load(cfg_path)
model = INRMark.load_from_checkpoint(ckpt_path, args=args).to(device).eval()

data_path = "/home/light_sun/workspace/inrsteg/data/DIV2K_valid"
imgs = os.listdir(data_path)
fixed_psnr = 38

ts = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])
coords = generate_grid_coordinates((-1, -1), 2, 128).unsqueeze(0).to(device)
total_acc = 0
total_psnr = 0
total_ssim = 0
n = 0
for img in tqdm(imgs):
    img_path = os.path.join(data_path, img)
    img = read_img(img_path)
    img = ts(img).unsqueeze(0).to(device)
    msg = gen_random_msg(args.msg_len).unsqueeze(0).to(device)
    # print(coords.size(), msg.size(), img.size())
    wm_img, mask = model.render_img(coords, msg, img)
    if fixed_psnr:
        wm_img = clip_psnr(wm_img, img, fixed_psnr, over_clip=True)

    predict_msg = model.decoder(wm_img)
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
