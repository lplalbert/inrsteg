import random
import torch

from FastTools.dataset.dataset import read_img
from FastTools.metre import PSNR
from FastTools.steganography.utils.common import msg_acc
from FastTools.util.TrainUtil import Args
import torch.nn.functional as F
import torch

def random_crop_tensor(image: torch.Tensor, crop_size: tuple) -> torch.Tensor: 
    _, height, width = image.shape
    crop_height, crop_width = crop_size
    left = random.randint(0, width - crop_width)
    top = random.randint(0, height - crop_height)
    right = left + crop_width
    bottom = top + crop_height
    return image[:, top:bottom, left:right]

def random_scale_tensor(image: torch.Tensor, min_scale: float, max_scale: float) -> torch.Tensor:
    """
    对输入张量图像进行随机缩放

    Args:
        image (torch.Tensor): 输入的图像张量，格式为 (C, H, W)
        min_scale (float): 最小缩放比例
        max_scale (float): 最大缩放比例

    Returns:
        torch.Tensor: 缩放后的图像张量
    """
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

def generate_coordinates(top_left: tuple, bottom_right: tuple, h: int, w: int) -> torch.Tensor:
    x1, y1 = top_left
    x2, y2 = bottom_right
    x_range = torch.linspace(x1, x2, steps=h)
    y_range = torch.linspace(y1, y2, steps=w)
    xx, yy = torch.meshgrid(x_range, y_range)
    coordinates = torch.stack((xx, yy), dim=-1)
    return coordinates

def image_to_blocks(image: torch.Tensor, global_coords: torch.Tensor, n: int, m: int) -> list:
    """
    将图像划分为n * m个块

    Args:
        image (torch.Tensor): 输入图像张量，形状为(C, H, W)，其中C是通道数，H是高度，W是宽度
        n (int): 垂直方向划分的块数
        m (int): 水平方向划分的块数

    Returns:
        list: 包含图像块的列表，每个元素是一个形状为(C, H // n, W // m)的张量
    """
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
    """
    将图像块重新组装成原始图像

    Args:
        blocks (list): 图像块列表
        n (int): 垂直方向划分的块数
        m (int): 水平方向划分的块数

    Returns:
        torch.Tensor: 重新组装后的图像张量，形状为(C, H, W)
    """
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

def load_model(ckpt, cfg="/home/light_sun/workspace/inrsteg/config/main.yaml"):
    from model.ismark_final_v3 import INRMark
    args = Args().load(cfg)
    model = INRMark.load_from_checkpoint(ckpt, args=args, map_location='cpu').eval()
    return model
    pass
if __name__ == "__main__":
    import torchvision
    k = 4
    full_img_size = 1024 * k
    num_block = 8 * k
    msg_len = 10
    
    # full_img_size = 128
    # num_block = 1
    
    msg = torch.randint(0, 2, (msg_len,)).float().unsqueeze(0)
    device = "cuda:3"
    
    print(msg)
    print(msg.size())
    with torch.no_grad():
        # coords = generate_coordinates((-1, -1), (1, 1), full_img_size, full_img_size)
        coords = generate_grid_coordinates((-1, -1), 2, full_img_size)
        model = load_model("/home/light_sun/workspace/inrsteg/output/ismark_final_v3_10/lightning_logs/version_0/checkpoints/ckpt-epoch=19-val_loss=0.1496.ckpt")
        ts = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((full_img_size, full_img_size))
        ])
        
        global_ts = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((128, 128)),
        ])
        
        
        img = read_img("/home/light_sun/workspace/inrsteg/data/DIV2K_train/0001.png")
        img = ts(img)
        
        # wm_block, _ = model.render_img(coords.unsqueeze(0), msg, img.unsqueeze(0))
        # predict_msg = model.decode_msg(wm_block)
        # acc = msg_acc(predict_msg, msg)
        # print(acc)
        
        # print(img.size())
        
        blocks, coords = image_to_blocks(img, coords, num_block, num_block)
 
 
        wm_blocks = []
        
        model = model.to(device).eval()
        for block, coord in zip(blocks, coords):
            block = block.unsqueeze(0)
            coord = coord.unsqueeze(0)
            
            coord = coord.to(device)
            msg = msg.to(device)
            block = block.to(device)
            wm_block, _ = model.render_img(coord, msg, block)
            # predict_msg = model.decode_msg(wm_block)
            # acc = msg_acc(predict_msg, msg) 
            # print(acc)          
            wm_block = wm_block.squeeze(0).cpu()
            wm_blocks.append(wm_block)
        
        
        wm_img = blocks_to_image(wm_blocks, num_block, num_block)
        # predict_msg = model.decode_msg(wm_img.unsqueeze(0).to(device))
        # acc = msg_acc(predict_msg, msg)   
        # print(acc)
        
        # wm_img = torch.clamp(wm_img, 0, 1)
        torchvision.utils.save_image(wm_img, "./test/test.png")
        torchvision.utils.save_image(10 * (wm_img-img), "./test/mask.png")
        psnr = PSNR(wm_img, img)
        print(psnr)
        
        # 计算嵌入信息
        # 随机裁剪
        # area = 0.01
        # ts = torchvision.transforms.Compose([
        #     torchvision.transforms.RandomResizedCrop((128, 128), scale=(area, area)),
        # ])
        
        crop_imgs = []
        for _ in range(20):
            crop_img = wm_img.clone()
            crop_img = random_scale_tensor(wm_img, 1, 1)
            # print(crop_img.size())
            crop_img = random_crop_tensor(crop_img, (128, 128))
            crop_imgs.append(crop_img)
            
        crop_imgs = torch.stack(crop_imgs).to(device)
        torchvision.utils.save_image(crop_imgs, "./test/test_crop.png")
        predict_msg = model.decode_msg(crop_imgs)
        msg = msg.expand(predict_msg.size(0), -1)
        acc = msg_acc(predict_msg, msg)
        print(acc)
        
    pass