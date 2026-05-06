import torch
import torch.nn as nn


class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.groups = groups
        
        # 调整通道数处理：输入输出通道数翻倍（实部+虚部）
        self.conv_layer = nn.Conv2d(
            in_channels=in_channels * 2,  # 处理实部和虚部
            out_channels=out_channels * 2,
            kernel_size=1,
            groups=self.groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.shape
        
        # 新版FFT处理
        # 步骤1：执行FFT变换
        fft = torch.fft.rfft2(x, norm='ortho')  # 输出复数张量 [B,C,H,W//2+1]
        
        # 分离实部和虚部
        real = fft.real  # [B,C,H,W//2+1]
        imag = fft.imag  # [B,C,H,W//2+1]
        
        # 拼接实部虚部作为通道维度
        fft_combined = torch.cat([real, imag], dim=1)  # [B, 2*C, H, W//2+1]
        
        # 步骤2：频域卷积处理
        fft_processed = self.conv_layer(fft_combined)  # [B, 2*out_C, H, W//2+1]
        fft_processed = self.relu(self.bn(fft_processed))
        
        # 拆分处理后的实部虚部
        real_new, imag_new = torch.chunk(fft_processed, 2, dim=1)  # 各[B, out_C, H, W//2+1]
        
        # 步骤3：逆FFT变换
        fft_new = torch.complex(real_new, imag_new)  # 重建复数张量
        output = torch.fft.irfft2(fft_new, s=(h, w), norm='ortho')  # [B, out_C, H, W]
        
        return output
    
class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class FFCNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=30):
        super().__init__()
        
        # Stem层：空间特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 输出: [B,32,224,224]

        # FFC模块堆叠
        self.block1 = FFC_BN_ACT(32, 64, kernel_size=3, 
                                ratio_gin=0.0,   # 输入无全局分量
                                ratio_gout=0.25, # 输出25%全局特征
                                stride=1,
                                padding=1)
        # 输出: local [B,48,224,224] + global [B,16,224,224]

        self.block2 = FFC_BN_ACT(64, 128, kernel_size=3,  # 总输入=48+16=64
                                ratio_gin=0.25, 
                                ratio_gout=0.25,
                                stride=2,       # 下采样
                                padding=1)
        # 输出: local [B,96,112,112] + global [B,32,112,112]

        self.block3 = FFC_BN_ACT(128, 256, kernel_size=3, # 总输入=96+32=128
                                ratio_gin=0.25,
                                ratio_gout=0.5,
                                stride=2,      # 下采样
                                padding=1)
        # 输出: local [B,128,56,56] + global [B,128,56,56]

        self.block4 = FFC_BN_ACT(256, 512, kernel_size=3, # 总输入=128+128=256
                                ratio_gin=0.5,
                                ratio_gout=0.75,
                                stride=2,      # 下采样
                                padding=1)
        # 输出: local [B,128,28,28] + global [B,384,28,28]

        # 特征合并与分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))  # 输出: [B,512,1,1]
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Stem层
        x = self.stem(x)  # [B,3,224,224] → [B,32,224,224]
        
        # FFC模块处理
        x = self.block1(x)  # 输入为单张量，输出tuple
        x = self.block2(x)  # 输入tuple，输出tuple
        x = self.block3(x)
        x = self.block4(x)
        
        # 合并特征
        local_feat, global_feat = x
        merged = torch.cat([local_feat, global_feat], dim=1)  # [B,512,28,28]
        
        # 分类头
        pooled = self.avgpool(merged).flatten(1)  # [B,512]
        return self.fc(pooled)  # [B,num_classes]
    
if __name__ == "__main__":
    import torch
    model = FFCNet(3)
    x = torch.randn(2, 3, 128, 128)
    print(model(x).size())
    
    pass