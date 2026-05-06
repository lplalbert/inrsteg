from random import random
import torch
import torchvision
from tqdm import tqdm

from FastTools.dataset.dataset import read_img
from FastTools.steganography.utils.common import msg_acc
from FastTools.util.TrainUtil import Args
from dataset.Mydataset import MyDataset, generate_grid_coordinates
from model.ismark_final_v3 import INRMark

device = 'cuda:0'
cfg = Args().load("/home/light_sun/workspace/inrsteg/config/main.yaml")
model = INRMark.load_from_checkpoint(
    "/home/light_sun/workspace/inrsteg/output/ismark_final_v3_10/lightning_logs/version_0/checkpoints/ckpt-epoch=14-val_loss=0.1054.ckpt",
    args = cfg
).eval().to(device)
dataset = MyDataset(cfg, data_len=10, valid=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
total_acc = 0
n = 0
ts = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomResizedCrop((128, 128))
        ])
img = read_img("/home/light_sun/workspace/inrsteg/data/DIV2K_train/0001.png")
img = ts(img).unsqueeze(0).to(device)
for batch in tqdm(dataloader):

    msg = batch['msg'].to(device)
    # img = batch['img'].to(device)
    # coords = batch['coords'].to(device)
    x = random() * (1-cfg.min_scale) 
    y = random() * (1-cfg.min_scale) 

    max_side = min(1-x, 1-y)
    side_length = 2 * (random() * (max_side - cfg.min_scale) + cfg.min_scale)
    x = x * 2 - 1
    y = y * 2 - 1
    
    grids = generate_grid_coordinates((x, y), side_length, grid_size=cfg.img_size).unsqueeze(0).to(device)
    wm_img, mask = model.render_img(grids, msg, img)
    
    predict_msg = model.decode_msg(wm_img)
    acc = msg_acc(predict_msg, msg)
    total_acc += acc.detach().cpu()
    n += 1
    # vutils().save_image(torch.cat([img, global_img], dim=0), "test.png")
    # break

print("acc: {}".format(total_acc / n))
