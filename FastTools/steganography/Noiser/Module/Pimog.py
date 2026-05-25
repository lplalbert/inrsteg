import time

import torch.nn as nn
import numpy as np
import torch
import kornia
import random
import math
from FastTools.steganography.Noiser.Noiser import NoiseLayer, NoiseLayerManager
import torch.nn.functional as F

def perspective(image_and_cover ,device,d=8, M=None):
    # the source points are the region to crop corners
    image, host_image = image_and_cover 

    c = image.shape[0]
    h = image.shape[2]
    w = image.shape[3]# destination size
    image_size = h
    if M == None:
        points_src=torch.ones(c,4,2)
        points_dst=torch.ones(c,4,2)
        for i in range(c):
            points_src[i,:,:] = torch.tensor([[
                [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.],
            ]])
            
            # the destination points are the image vertexes   
            # d=8   
            tl_x = random.uniform(-d, d)     # Top left corner, top
            tl_y = random.uniform(-d, d)    # Top left corner, left
            bl_x = random.uniform(-d, d)   # Bot left corner, bot
            bl_y = random.uniform(-d, d)    # Bot left corner, left
            tr_x = random.uniform(-d, d)     # Top right corner, top
            tr_y = random.uniform(-d, d)   # Top right corner, right
            br_x = random.uniform(-d, d)  # Bot right corner, bot
            br_y = random.uniform(-d, d)   # Bot right corner, right
            
            points_dst[i,:,:] = torch.tensor([[
                [tl_x, tl_y],
                [tr_x + image_size, tr_y],
                [br_x + image_size, br_y + image_size],
                [bl_x, bl_y +  image_size],
            ]])
        # compute perspective transform
        M: torch.tensor = kornia.geometry.get_perspective_transform(points_src, points_dst).to(device)
    
    # warp the original image by the found transform
    data_warp: torch.tensor = kornia.geometry.warp_perspective(image.float(), M, dsize=(h, w)).to(device)
    host_warp: torch.tensor = kornia.geometry.warp_perspective(host_image.float(), M, dsize=(h, w)).to(device) 
    return data_warp, host_warp, M

def MoireGen(p_size, theta, center_x, center_y):
    # z = np.zeros((p_size, p_size))
    # for i in range(p_size):
    #     for j in range(p_size):
    #         z1 = 0.5+0.5*math.cos(2*math.pi*np.sqrt((i+1-center_x)**2+(j+1-center_y)**2))
    #         z2 = 0.5+0.5*math.cos(math.cos(theta/180*math.pi)*(j+1)+math.sin(theta/180*math.pi)*(i+1))
    #         z[i,j] = np.min([z1,z2])
    # M = (z+1)/2
    # return M
    # 创建包含坐标(i, j)的数组
    i, j = np.meshgrid(np.arange(p_size), np.arange(p_size), indexing='ij')

    # 计算距离中心(center_x, center_y)的距离
    distance_from_center = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)

    # 使用向量化运算计算 z1 和 z2
    z1 = 0.5 + 0.5 * np.cos(2 * np.pi * distance_from_center)
    z2 = 0.5 + 0.5 * np.cos(np.cos(theta / 180 * np.pi) * j + np.sin(theta / 180 * np.pi) * i)

    # 计算数组中每个元素的最小值，这里同样使用向量化运算
    z = np.minimum(z1, z2)

    # 使用向量化运算计算 M
    M = (z + 1) / 2

    return M

def Light_Distortion(c,embed_image):
    mask = np.zeros((embed_image.shape))
    mask_2d = np.zeros((embed_image.shape[2],embed_image.shape[3]))
    a = 0.7+np.random.rand(1)*0.2
    b = 1.1+np.random.rand(1)*0.2
    if c == 0:
        direction = np.random.randint(1,5)
        for i in range(embed_image.shape[2]):
            mask_2d[i,:] = -((b-a)/(mask.shape[2]-1))*(i-mask.shape[3])+a
        if direction == 1:
            O = mask_2d
        elif direction == 2:
            O = np.rot90(mask_2d,1)
        elif direction == 3:
            O = np.rot90(mask_2d,2)
        elif direction == 4:
            O = np.rot90(mask_2d,3)
        # for batch in range(embed_image.shape[0]):
        # 	for channel in range(embed_image.shape[1]):
        # 		mask[batch,channel,:,:] = mask_2d
    else:
        x = np.random.randint(0,mask.shape[2])
        y = np.random.randint(0,mask.shape[3])
        max_len = np.max([np.sqrt(x**2+y**2),np.sqrt((x-255)**2+y**2),np.sqrt(x**2+(y-255)**2),np.sqrt((x-255)**2+(y-255)**2)])
        # for i in range(mask.shape[2]):
        #     for j in range(mask.shape[3]):
        #         mask[:,:,i,j] = np.sqrt((i-x)**2+(j-y)**2)/max_len*(a-b)+b
        i, j = np.meshgrid(np.arange(embed_image.shape[2]), np.arange(embed_image.shape[3]), indexing='ij')
        distances = np.sqrt((i - x) ** 2 + (j - y) ** 2)
        mask = (distances / max_len) * (a - b) + b
        mask = np.tile(mask, (embed_image.shape[0], embed_image.shape[1], 1, 1))
        O = mask
    return O

def Moire_Distortion(embed_image):
    Z = np.zeros((embed_image.shape))
    for i in range(3):
        theta = np.random.randint(0,180)
        center_x = np.random.rand(1)*embed_image.shape[2]
        center_y = np.random.rand(1)*embed_image.shape[3]
        M = MoireGen(embed_image.shape[2], theta, center_x, center_y)
        Z[:,i,:,:] = M
    return Z

class ScreenShooting(nn.Module):

    def __init__(self):
        super(ScreenShooting, self).__init__()
        self.noise_type = 'PIMOG'
        self.last_M = None
        self.last_L = None
        self.last_Z = None
        self.last_noise = None
        

    def forward(self, embed_image, repeat=False):
        if len(embed_image) == 1: 
            embed_image = [embed_image, embed_image.clone()] 
        
        embed_image, host_img = embed_image[0], embed_image[1]
        noised_image = torch.zeros_like(embed_image)
        device = embed_image.device

        # perspective transform
        if repeat:
            noised_image, host_img, M = perspective([embed_image, host_img],device, 8, self.last_M)
        else:
            noised_image, host_img, M = perspective([embed_image, host_img],device,8)
            self.last_M = M

        # Light Distortion
        if repeat:
            L = self.last_L
        else:
            c = np.random.randint(0,2)
            L = Light_Distortion(c,embed_image)
            self.last_L = L

        # Moire Distortion
        if repeat:
            Z = self.last_Z
        else:
            Z = Moire_Distortion(embed_image)*2-1
            self.last_Z = Z
        
        Mo = Z.copy()
        Li = L.copy()

        
        noised_image = noised_image*torch.from_numpy(Li).to(device)*0.85+torch.from_numpy(Mo).to(device)*0.15
        
        #Gaussian noise
        if repeat:
            random_noise = self.last_noise
        else:
            random_noise = 0.001**0.5*torch.randn(noised_image.size()).to(device)
            self.last_noise = random_noise
        noised_image = noised_image + random_noise
        
        
        return noised_image.float(), host_img.float()


@NoiseLayerManager.register('Pimog')
class Pimog():
    def __init__(self):
        super().__init__()
        self.model = ScreenShooting()
        pass

    def noise(self, img, cover_img, repeat=False):
        b, c, h, w = img.size()
        img = img * 2 - 1
        cover_img = cover_img * 2 - 1
        if h != w:
            size = max(h, w)
            img = F.interpolate(img, size=(size, size), mode='bilinear', align_corners=True)
            cover_img = F.interpolate(cover_img, size=(size, size), mode='bilinear', align_corners=True)
        img, cover_img = self.model([img, cover_img], repeat)
        if h != w:
            img = F.interpolate(img, size=(h, w), mode='bilinear', align_corners=True)
            cover_img = F.interpolate(cover_img, size=(h, w), mode='bilinear', align_corners=True)
        
        img = (img + 1) / 2
        cover_img = (cover_img + 1) / 2
        
        return img, cover_img
        pass

    def latest_noise(self, img, cover_img):
        return self.noise(img, cover_img, repeat=True)
    pass

# class Identity(nn.Module):

#     def __init__(self):
#         super(Identity, self).__init__()

#     def forward(self, embed_image):
#         output = embed_image
#         return output

if __name__ == '__main__':
    # img = torch.randn(1,3,256, 256)
    from PIL import Image 
    import torchvision

    img = Image.open('/mnt/xsj2023/Datasets/crop_data/val/000002.png') 
    h, w= img.size 
    img = torchvision.transforms.Resize((512, 512))(img)
    img = torchvision.transforms.ToTensor()(img) 
    img = img.unsqueeze(0) 
    scr = ScreenShooting()
    noise_img = scr([img , None]) 
    # save the image 
    torchvision.utils.save_image(noise_img, 'noise_img.jpg') 
