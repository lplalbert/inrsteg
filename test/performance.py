import random
import torch
from tqdm import tqdm

from FastTools.dataset.dataset import read_img
from FastTools.metre import PSNR
from FastTools.metric.ssim import SSIM
from FastTools.steganography.utils.common import msg_acc
from FastTools.util.TrainUtil import Args
import torch.nn.functional as F
import torch
import os
from PIL import Image
import torchvision
from torchvision import transforms

def random_crop_tensor(image: torch.Tensor, crop_size: tuple) -> torch.Tensor:
    _, height, width = image.shape
    crop_height, crop_width = crop_size
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    return image[:, top:bottom, left:right]




def random_scale_tensor(image: torch.Tensor, min_scale: float, max_scale: float) -> torch.Tensor:
    scale_factor = random.uniform(min_scale, max_scale)
    _, height, width = image.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    return F.interpolate(image.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)


def generate_grid_coordinates(top_left, side_length, grid_size=128):
    """
    生成固定大小的坐标矩阵
    
    :param top_left: 左上角点的坐标 (x1, y1)
    :param side_length: 矩阵的边长
    :param grid_size: 坐标矩阵的大小 (默认 128x128)
    :return: 坐标矩阵 [grid_size, grid_size, 2]
    """
    # 生成网格坐标
    x = torch.linspace(top_left[0], top_left[0] + side_length, grid_size)
    y = torch.linspace(top_left[1], top_left[1] + side_length, grid_size)
    
    # 生成坐标矩阵
    xv, yv = torch.meshgrid(x, y, indexing='ij')
    coordinates = torch.stack([xv, yv], dim=-1)
    
    return coordinates





def image_to_blocks(image: torch.Tensor, global_coords: torch.Tensor, n: int, m: int) -> list:
    _, height, width = image.shape
    block_height = height // n
    block_width = width // m
    blocks = []
    coords = []
    for i in range(n):
        for j in range(m):
            block = image[:, i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width]
            coord = global_coords[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width, :]
            blocks.append(block)
            coords.append(coord)
    return blocks, coords


def blocks_to_image(blocks: list, n: int, m: int) -> torch.Tensor:
    block = blocks[0]
    _, block_height, block_width = block.shape
    height = n * block_height
    width = m * block_width
    channels = block.shape[0]
    image = torch.zeros((channels, height, width))
    for i in range(n):
        for j in range(m):
            block_index = i * m + j
            image[:, i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width] = blocks[block_index]
    return image

def load_model(ckpt, cfg="/home/sn/workspace/inrsteg/config/main.yaml"):
    from model.ismark_final_v12 import INRMark
    args = Args().load(cfg)
    model = INRMark.load_from_checkpoint(ckpt, args=args, map_location='cpu').eval()
    return model
    pass


def valid_performance(data_folder, model, noise_fn, msg_len=30, img_size=2048, num_block=None, render_device="cuda:0"):
    with torch.no_grad():
        model.eval().to(render_device)
        if num_block is None:
            num_block = img_size // 1024 * 8
        ts = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((img_size, img_size))
        ])
        global_coords = generate_grid_coordinates((-1, -1), 2, img_size)
        imgs = os.listdir(data_folder)
        
        total_acc = 0
        total_psnr = 0
        total_ssim = 0
        n = 0
        for name in tqdm(imgs):
            img_path = os.path.join(data_folder, name)
            img = Image.open(img_path).convert("RGB")
            img = ts(img)
            global_msg = torch.randint(0, 2, (msg_len,)).float()
            # 分块渲染
            blocks, coords = image_to_blocks(img, global_coords, num_block, num_block)
            wm_blocks = []
            for block, coord in zip(blocks, coords):
                block = block.unsqueeze(0)
                coord = coord.unsqueeze(0)
                msg = global_msg.unsqueeze(0)
                
                coord = coord.to(render_device)
                msg = msg.to(render_device)
                block = block.to(render_device)
                wm_block, _ = model.render_img(coord, msg, block)
                wm_block = wm_block.squeeze(0).cpu()
                wm_blocks.append(wm_block)
            wm_img = blocks_to_image(wm_blocks, num_block, num_block) # 单张图片无batch
            
            # 施加噪声
            noised_wm_img = noise_fn(wm_img)
            # 预测信息
            predict_msg = model.decode_msg(noised_wm_img.to(render_device)).cpu()
            global_msg = global_msg.unsqueeze(0).expand(predict_msg.size(0), -1)
            
            # 计算ACC, PSNR, SSIM等信息
            acc = msg_acc(predict_msg.to(render_device), global_msg.to(render_device))
            psnr = PSNR(wm_img.unsqueeze(0).to(render_device), img.unsqueeze(0).to(render_device))
            ssim = SSIM(wm_img.unsqueeze(0).to(render_device), img.unsqueeze(0).to(render_device))
            
            total_acc += acc.cpu().item()
            total_psnr += psnr.cpu().item()
            total_ssim += ssim.cpu().item()
            n += 1
            pass
        
        print("ACC: {:.4f}".format(total_acc / n))
        print("PSNR: {:.4f}".format(total_psnr / n))
        print("SSIM: {:.4f}".format(total_ssim / n))
        pass
    pass

if __name__ == "__main__":
    
    def noise(img):
        # 随机裁剪128x128的块
        # img = random_crop_tensor(img, (128, 128)).unsqueeze(0)
        # 缩放到128x128的尺寸上
        img = F.interpolate(img.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False)
        return img
        pass
    
    
    
    ckpt_path = "/home/sn/workspace/inrsteg/output/ismark_final_v12/lightning_logs/version_0/checkpoints/ckpt-epoch=1574-val_loss=0.0089.ckpt"
    data_folder_path = "/data/sn/datasets/DIV2K/DIV2K_valid_HR"
    model = load_model(
        ckpt_path
    )
    valid_performance(
        data_folder_path,
        model,
        noise
    )
    
    # import torchvision
    # k = 2
    # full_img_size = 1024 * k
    # num_block = 8 * k
    # msg_len = 30
    
    # # full_img_size = 128
    # # num_block = 1
    
    # msg = torch.randint(0, 2, (msg_len,)).float().unsqueeze(0)
    # device = "cuda:3"
    
    # print(msg)
    # print(msg.size())
    # with torch.no_grad():
    #     # coords = generate_coordinates((-1, -1), (1, 1), full_img_size, full_img_size)
    #     coords = generate_grid_coordinates((-1, -1), 2, full_img_size)
    #     model = load_model("/home/sn/workspace/inrsteg/output/ismark_final_v12/lightning_logs/version_0/checkpoints/ckpt-epoch=129-val_loss=0.0158.ckpt")
    #     ts = torchvision.transforms.Compose([
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Resize((full_img_size, full_img_size))
    #     ])
        
    #     global_ts = torchvision.transforms.Compose([
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Resize((128, 128)),
    #     ])
        
        
    #     img = read_img("/data/sn/datasets/DIV2K/DIV2K_train_HR/0001.png")
    #     img = ts(img)
        
    #     # wm_block, _ = model.render_img(coords.unsqueeze(0), msg, img.unsqueeze(0))
    #     # predict_msg = model.decode_msg(wm_block)
    #     # acc = msg_acc(predict_msg, msg)
    #     # print(acc)
        
    #     # print(img.size())
        
    #     blocks, coords = image_to_blocks(img, coords, num_block, num_block)
    #     wm_blocks = []
        
    #     model = model.to(device).eval()
    #     for block, coord in zip(blocks, coords):
    #         block = block.unsqueeze(0)
    #         coord = coord.unsqueeze(0)
            
    #         coord = coord.to(device)
    #         msg = msg.to(device)
    #         block = block.to(device)
    #         wm_block, _ = model.render_img(coord, msg, block)
    #         # predict_msg = model.decode_msg(wm_block)
    #         # acc = msg_acc(predict_msg, msg) 
    #         # print(acc)          
    #         wm_block = wm_block.squeeze(0).cpu()
    #         wm_blocks.append(wm_block)
        
        
    #     wm_img = blocks_to_image(wm_blocks, num_block, num_block)
        
 
    #     wm_blocks = []
        
    #     model = model.to(device).eval()
    #     for block, coord in zip(blocks, coords):
    #         block = block.unsqueeze(0)
    #         coord = coord.unsqueeze(0)
            
    #         coord = coord.to(device)
    #         msg = msg.to(device)
    #         block = block.to(device)
    #         wm_block, _ = model.render_img(coord, msg, block)
    #         # predict_msg = model.decode_msg(wm_block)
    #         # acc = msg_acc(predict_msg, msg) 
    #         # print(acc)          
    #         wm_block = wm_block.squeeze(0).cpu()
    #         wm_blocks.append(wm_block)
        
        
    #     wm_img = blocks_to_image(wm_blocks, num_block, num_block)
    #     # predict_msg = model.decode_msg(wm_img.unsqueeze(0).to(device))
    #     # acc = msg_acc(predict_msg, msg)   
    #     # print(acc)
        
    #     # wm_img = torch.clamp(wm_img, 0, 1)
    #     torchvision.utils.save_image(wm_img, "./test/test.png")
    #     torchvision.utils.save_image(10 * (wm_img-img), "./test/mask.png")
    #     psnr = PSNR(wm_img, img)
    #     print(psnr)
        
    #     # 计算嵌入信息
    #     # 随机裁剪
    #     # area = 0.01
    #     # ts = torchvision.transforms.Compose([
    #     #     torchvision.transforms.RandomResizedCrop((128, 128), scale=(area, area)),
    #     # ])
        
    #     crop_imgs = []
    #     for _ in range(20):
    #         crop_img = wm_img.clone()
    #         crop_img = random_scale_tensor(wm_img, 1, 1)
    #         # print(crop_img.size())
    #         crop_img = random_crop_tensor(crop_img, (128, 128))
    #         crop_imgs.append(crop_img)
            
    #     crop_imgs = torch.stack(crop_imgs).to(device)
    #     torchvision.utils.save_image(crop_imgs, "./test/test_crop.png")
    #     predict_msg = model.decode_msg(crop_imgs)
    #     msg = msg.expand(predict_msg.size(0), -1)
    #     acc = msg_acc(predict_msg, msg)
    #     print(acc)
        
    # pass