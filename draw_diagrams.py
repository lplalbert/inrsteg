"""
INRMark 架构图 — 修复对齐 + 中文
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── 字体 ──
for _p in ['/tmp/NotoSansCJK-SC-Regular.ttf', '/tmp/NotoSansCJK-SC-Bold.ttf']:
    try: fm.fontManager.addfont(_p)
    except: pass
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def box(ax, cx, cy, w, h, text='', fc='#EEF0FA', ec='#667eea', lw=2, fs=11, fw='normal', tc='#222'):
    """以中心点(cx,cy)画圆角矩形"""
    x, y = cx - w/2, cy - h/2
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                       fc=fc, ec=ec, linewidth=lw, zorder=2)
    ax.add_patch(p)
    if text:
        lines = text.split('\n')
        n = len(lines)
        for i, line in enumerate(lines):
            sz = fs - 1 if i > 0 else fs
            wt = 'normal' if i > 0 else fw
            cl = '#555' if i > 0 else tc
            yoff = (i - (n-1)/2) * (fs + 3)
            ax.text(cx, cy - yoff, line, ha='center', va='center',
                    fontsize=sz, fontweight=wt, color=cl, zorder=3)


def arr(ax, x1, y1, x2, y2, color='#666', lw=1.8, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle=ls),
                zorder=1)


def fig_save(fig, name):
    fig.savefig(f'/mnt/lpl/inrsteg-final_v1/{name}', dpi=200,
                bbox_inches='tight', facecolor='#f8f9fc')
    plt.close()
    print(f'{name} ok')


# ════════════════════════════════════════════════════════════
# 图1: 整体架构
# ════════════════════════════════════════════════════════════
def draw_overview():
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18); ax.set_ylim(0, 11)
    ax.axis('off'); ax.set_facecolor('#f8f9fc')

    # 标题
    ax.text(9, 10.5, 'INRMark 整体架构', ha='center', fontsize=22, fontweight='bold', color='#1a1a2e')
    ax.text(9, 10.05, '编码器 (坐标驱动INR) + 解码器 (CNN) + 噪声层  |  f(坐标, 消息) = 水印残差',
            ha='center', fontsize=12, color='#666')

    # ── 编码器背景 ──
    enc_bg = FancyBboxPatch((0.5, 3.2), 7.5, 6.5, boxstyle='round,pad=0.3',
                            fc='white', ec='#667eea', lw=2.5, zorder=0)
    ax.add_patch(enc_bg)
    ax.text(4.25, 9.35, '编码器 — INR (隐式神经表示)', ha='center',
            fontsize=15, fontweight='bold', color='#667eea')

    # coords
    box(ax, 4.25, 8.7, 5.0, 0.65, '坐标 coords [B, 128, 128, 2]   范围 [-1, 1]²',
        '#EDEBFF', '#667eea', fs=12)
    arr(ax, 4.25, 8.37, 4.25, 8.1)

    # FeatureGrid
    box(ax, 4.25, 7.35, 6.5, 1.2,
        'FeatureGrid 结构编码器\n8层多分辨率网格 (256²→2²) + 双线性采样\nFourierFeatMapping sin/cos 高频编码  →  struct_feat [B, N, 256]',
        '#EDEBFF', '#667eea', fs=11)
    arr(ax, 4.25, 6.75, 4.25, 6.45)

    # LowRank
    box(ax, 4.25, 5.65, 6.5, 1.3,
        '4× LowRankFusionBlock\n低秩双线性融合: x_fc(257→30) ⊙ y_fc(129→30)\nout_fc(30→256) + 残差连接  |  rank_dim=30',
        '#E3F0FF', '#4facfe', fs=11)

    # predict
    arr(ax, 4.25, 5.0, 4.25, 4.7)
    box(ax, 4.25, 4.35, 3.5, 0.6, 'predict: Linear(256→3)', '#E5F9E5', '#43e97b', fs=12)

    # wm output
    arr(ax, 4.25, 4.05, 4.25, 3.7)

    # ── 消息路径 ──
    box(ax, 13.5, 8.7, 3.5, 0.65, '消息 msg [B, 30]', '#FFF0F5', '#c471ed', fs=12)
    arr(ax, 13.5, 8.37, 13.5, 8.1)
    box(ax, 13.5, 7.35, 3.5, 1.2,
        'msg_encoder\nMLP: 30→128→128→128\nBN+ReLU  →  msg_feat [B,128]',
        '#FFF0F5', '#c471ed', fs=11)

    # msg → LowRank 虚线
    arr(ax, 11.75, 7.0, 7.5, 5.65, color='#c471ed', lw=1.5)
    ax.text(9.5, 6.7, 'expand→[B×N,128]', fontsize=9, color='#888', ha='center')

    # ── 输出 wm ──
    box(ax, 4.25, 2.85, 6.5, 0.9,
        'wm_img = residual × α + img   clamp(-1,1) → clamp(0,1)   [B, 3, 128, 128]',
        '#FFF5E5', '#f6d365', fs=11)
    arr(ax, 4.25, 3.40, 4.25, 3.30, color='#f6d365', lw=2)

    # ── 解码器背景 ──
    dec_bg = FancyBboxPatch((9.0, 0.5), 8.5, 4.0, boxstyle='round,pad=0.3',
                            fc='white', ec='#f5576c', lw=2.5, zorder=0)
    ax.add_patch(dec_bg)
    ax.text(13.25, 4.2, '解码器 — HiddenDecoder (CNN)', ha='center',
            fontsize=15, fontweight='bold', color='#f5576c')

    box(ax, 13.25, 3.6, 7.5, 0.6,
        'ConvBNRelu(3→128)  |  3× [SEBlock + Conv2d(k4,s2)]   128²→63²→30²→14²',
        '#FFEDED', '#f5576c', fs=11)
    arr(ax, 13.25, 3.3, 13.25, 3.05)
    box(ax, 13.25, 2.7, 6.5, 0.6,
        'Conv(128→90) repeat=3  →  AvgPool(1) → Linear(90→30)',
        '#FFEDED', '#f5576c', fs=11)
    arr(ax, 13.25, 2.4, 13.25, 2.15)
    box(ax, 13.25, 1.75, 4.0, 0.7, 'pred_msg [B, 30]', '#FFD5D5', '#f5576c', fs=13, fw='bold')

    # wm → decoder 虚线
    arr(ax, 7.5, 2.85, 11.25, 1.75, color='#f5576c', lw=2)
    ax.text(9.0, 2.6, 'noised', fontsize=9, color='#f5576c', ha='center')

    # ── Noiser ──
    box(ax, 4.25, 1.35, 6.5, 1.2,
        'Noiser 噪声层 (12种攻击)\n旋转 / 裁剪 / 平移 / 缩放 / 剪切\nJPEG / 高斯噪声 / 色彩抖动 / ...',
        '#FFF8E5', '#f6d365', fs=10)

    # ── Loss ──
    box(ax, 9, 0.15, 17.0, 0.55,
        'Loss = w_msg · MSE(pred_msg, msg)  +  w_img · MSE(wm_img, img)  +  w_lpips · LPIPS',
        '#F0F0F0', '#999', fs=12, fw='bold')

    fig_save(fig, 'diagram1_overview.png')


# ════════════════════════════════════════════════════════════
# 图2: FeatureGrid
# ════════════════════════════════════════════════════════════
def draw_featuregrid():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18); ax.set_ylim(0, 10)
    ax.axis('off'); ax.set_facecolor('#f8f9fc')

    ax.text(9, 9.5, 'FeatureGrid — 多分辨率结构编码器', ha='center',
            fontsize=22, fontweight='bold', color='#1a1a2e')
    ax.text(9, 9.05, '给一个坐标 (x,y)，返回该点的 256 维结构特征向量',
            ha='center', fontsize=12, color='#666')

    # 输入
    box(ax, 9, 8.35, 7.0, 0.6, '坐标 coords [B, 128, 128, 2]   范围 [-1, 1]²',
        '#EDEBFF', '#667eea', fs=13)
    arr(ax, 9, 8.05, 9, 7.8)

    # 8层网格 — 均匀排列
    names = ['Grid 0 [32,256,256]', 'Grid 1 [32,128,128]', 'Grid 2 [32,64,64]',
             'Grid 3 [32,32,32]', 'Grid 4 [32,16,16]', 'Grid 5 [32,8,8]',
             'Grid 6 [32,4,4]', 'Grid 7 [32,2,2]']
    gw = 1.9
    total_w = 8 * gw + 7 * 0.15
    x_start = 9 - total_w / 2 + gw / 2

    for i, nm in enumerate(names):
        cx = x_start + i * (gw + 0.15)
        fc = '#EDEBFF' if i < 4 else '#F3F2FA'
        ec = '#667eea' if i < 4 else '#aaa'
        box(ax, cx, 7.1, gw, 0.7, nm, fc, ec, fs=9, lw=1.2)

    ax.text(x_start, 6.6, '← 最细: 局部细节', fontsize=9, color='#667eea', ha='center')
    ax.text(x_start + 7 * (gw + 0.15), 6.6, '最粗: 全局 →', fontsize=9, color='#999', ha='center')

    # bilinear sampling
    arr(ax, 9, 6.75, 9, 6.3)
    box(ax, 9, 5.8, 15.0, 0.65,
        'grid_sample(grid, coord, mode="bilinear")   每个坐标在8层网格上双线性插值 → 8 × 32 = 256 维',
        '#EDEBFF', '#667eea', fs=12)

    # Fourier 路径
    box(ax, 15, 8.35, 4.5, 0.6, 'FourierFeatMapping', '#F5F0FF', '#764ba2', fs=12)
    box(ax, 15, 7.55, 4.5, 0.7,
        'x_proj = coords @ B\ncat(sin,cos) → 16维',
        '#F5F0FF', '#764ba2', fs=10)
    arr(ax, 13.5, 8.05, 12.75, 8.05, color='#764ba2', lw=1.5)
    arr(ax, 15, 7.2, 15, 6.15, color='#764ba2', lw=1.5)

    # concat
    arr(ax, 9, 5.47, 9, 5.1)
    box(ax, 9, 4.7, 13.0, 0.6, 'concat 拼接 → [B, N, 272]   (256 网格特征 + 16 傅里叶编码)',
        '#E8E5F5', '#667eea', fs=12)

    # adapter
    arr(ax, 9, 4.4, 9, 4.05)
    box(ax, 9, 3.7, 12.0, 0.6, 'adapter: Linear(272→256) → BatchNorm → SiLU',
        '#E8E5F5', '#667eea', fs=12)

    # out
    arr(ax, 9, 3.4, 9, 3.05)
    box(ax, 9, 2.7, 12.0, 0.6, 'out: Linear(256→256) → BatchNorm → SiLU',
        '#E8E5F5', '#667eea', fs=12)

    # 设计原理
    box(ax, 9, 1.2, 16.0, 1.6,
        '设计原理\n'
        '粗网格捕获全局区域  |  细网格捕获局部纹理  |  多分辨率融合\n'
        'Fourier编码: 显式注入高频, 克服MLP光谱偏差\n'
        '连续坐标: 任意精度采样, 无像素离散化约束',
        '#FFF', '#ddd', fs=11, lw=1)

    fig_save(fig, 'diagram2_featuregrid.png')


# ════════════════════════════════════════════════════════════
# 图3: LowRankFusionBlock
# ════════════════════════════════════════════════════════════
def draw_lowrank():
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18); ax.set_ylim(0, 11)
    ax.axis('off'); ax.set_facecolor('#f8f9fc')

    ax.text(9, 10.5, 'LowRankFusionBlock — 低秩双线性融合 (核心创新)', ha='center',
            fontsize=22, fontweight='bold', color='#1a1a2e')
    ax.text(9, 10.05, '参数量仅 19K (对比朴素拼接 98K，减少 80%)',
            ha='center', fontsize=12, color='#666')

    # ── 输入 ──
    box(ax, 4, 9.3, 5.0, 0.8, 'struct_feat [B×N, 256]\n+ ones → [B×N, 257]',
        '#EDEBFF', '#667eea', fs=12)
    box(ax, 14, 9.3, 5.0, 0.8, 'msg_feat [B×N, 128]\n+ ones → [B×N, 129]',
        '#FFF0F5', '#c471ed', fs=12)

    arr(ax, 4, 8.9, 4, 8.55)
    arr(ax, 14, 8.9, 14, 8.55)

    # ── x_fc / y_fc ──
    box(ax, 4, 8.05, 5.0, 0.7, 'x_fc  Linear(257→30)  无bias\n结构特征降维到秩空间',
        '#EDEBFF', '#667eea', fs=11)
    box(ax, 14, 8.05, 5.0, 0.7, 'y_fc  Linear(129→30)  无bias\n消息特征降维到秩空间',
        '#FFF0F5', '#c471ed', fs=11)

    arr(ax, 4, 7.7, 4, 7.2)
    arr(ax, 14, 7.7, 14, 7.2)

    # ── Hadamard 中心 ──
    had = FancyBboxPatch((3, 6.45), 12, 1.1, boxstyle='round,pad=0.2',
                         fc='#E3F0FF', ec='#4facfe', lw=2.5, zorder=2)
    ax.add_patch(had)
    ax.text(9, 7.2, 'z = x ⊙ y  (逐元素 Hadamard 积)', ha='center',
            fontsize=17, fontweight='bold', color='#2a6a9a', zorder=3)
    ax.text(9, 6.7, '[B×N, 30]  双线性池化 — 每个秩维度 = 一个结构×消息交互',
            ha='center', fontsize=11, color='#555', zorder=3)

    # 连线到 Hadamard
    ax.plot([4, 4, 3.5], [7.7, 7.2, 7.2], color='#666', lw=1.8, zorder=1)
    ax.plot([14, 14, 14.5], [7.7, 7.2, 7.2], color='#666', lw=1.8, zorder=1)

    arr(ax, 9, 6.45, 9, 5.95)

    # ── out_fc ──
    box(ax, 9, 5.5, 12.0, 0.7, 'out_fc  Linear(30→256) → BatchNorm → ReLU  无bias',
        '#E3F0FF', '#4facfe', fs=12)

    arr(ax, 9, 5.15, 9, 4.75)

    # ── 残差连接 ──
    ax.plot([1.5, 1.5, 4.0], [7.7, 4.75, 4.75], color='#43e97b', lw=2.5,
            linestyle='--', zorder=1)
    ax.text(1.3, 6.2, '残差\n连接', fontsize=11, color='#2a7a2a', ha='center', va='center')
    ax.annotate('', xy=(4.0, 4.75), xytext=(2.0, 4.75),
                arrowprops=dict(arrowstyle='->', color='#43e97b', lw=2.5), zorder=1)

    # ── 输出 ──
    box(ax, 9, 4.2, 12.0, 0.8, 'output = raw_x + z  →  [B×N, 256]\n残差连接: 原始结构特征不丢失',
        '#D5F0FF', '#4facfe', fs=14, fw='bold')

    # ── 4×串联 ──
    box(ax, 9, 2.8, 16.0, 1.8,
        '4× 串联:  Block 0  →  Block 1  →  Block 2  →  Block 3\n\n'
        'msg_feat 恒定注入所有 Block，struct_feat 逐步被"消息化"\n'
        '每个 Block 参数独立，共享相同结构但不共享权重',
        '#EBF5FF', '#4facfe', fs=12)

    # ── Why Low-Rank ──
    box(ax, 9, 0.9, 16.0, 1.0,
        '为什么用"低秩"?    朴素拼接: Linear(384→256) = 98K 参数   |   低秩融合: 19K 参数 (减少 80%)\n'
        'Rank=30=msg_len, 每个秩维度对应一个消息比特的结构交互',
        '#FFF', '#ddd', fs=11, lw=1)

    fig_save(fig, 'diagram3_lowrank.png')


# ════════════════════════════════════════════════════════════
# 图4: HiddenDecoder
# ════════════════════════════════════════════════════════════
def draw_decoder():
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18); ax.set_ylim(0, 11)
    ax.axis('off'); ax.set_facecolor('#f8f9fc')

    ax.text(9, 10.5, 'HiddenDecoder — CNN 解码器', ha='center',
            fontsize=22, fontweight='bold', color='#1a1a2e')
    ax.text(9, 10.05, '从水印图中提取 30bit 消息  |  空间压缩: 128² → 14² → 30',
            ha='center', fontsize=12, color='#666')

    # 输入
    box(ax, 9, 9.35, 8.0, 0.65, '水印图 / noised_img   [B, 3, 128, 128]',
        '#FFEDED', '#f5576c', fs=14, fw='bold')
    arr(ax, 9, 9.02, 9, 8.75)

    # Layer 0
    box(ax, 9, 8.3, 10.0, 0.65, 'ConvBNRelu(3→128) k=3 s=1 p=1   →   [B, 128, 128, 128]   浅层特征, 保持分辨率',
        '#FFEDED', '#f5576c', fs=12)
    arr(ax, 9, 7.97, 9, 7.6)

    # 3 blocks — 水平均匀排列
    bw = 4.8
    gap = 0.3
    total = 3 * bw + 2 * gap
    sx = 9 - total / 2 + bw / 2
    block_y = 6.5
    block_h = 1.8

    blocks = [
        ('Block 1', 'ConvBNRelu(128→128)\nSEBlock(128) 通道注意力\nConv2d(k4,s2,p0) 下采样', '128 → 63  (×2)'),
        ('Block 2', 'ConvBNRelu(128→128)\nSEBlock(128) 通道注意力\nConv2d(k4,s2,p0) 下采样', '63 → 30'),
        ('Block 3', 'ConvBNRelu(128→128)\nSEBlock(128) 通道注意力\nConv2d(k4,s2,p0) 下采样', '30 → 14'),
    ]

    for i, (name, desc, size) in enumerate(blocks):
        cx = sx + i * (bw + gap)
        box(ax, cx, block_y, bw, block_h, f'{name}\n{desc}',
            '#FFEDED', '#f5576c', fs=11)
        ax.text(cx, block_y - block_h/2 - 0.15, size, ha='center',
                fontsize=10, color='#d4475c')

    # 块间箭头
    for i in range(2):
        x1 = sx + i * (bw + gap) + bw/2
        x2 = sx + (i+1) * (bw + gap) - bw/2
        arr(ax, x1 + bw/2 + 0.05, block_y, x2 - bw/2 + bw - 0.05, block_y)

    arr(ax, 9, block_y - block_h/2 - 0.35, 9, block_y - block_h/2 - 0.75)

    # SEBlock 详解
    se = FancyBboxPatch((1.5, 4.1), 15.0, 0.9, boxstyle='round,pad=0.15',
                        fc='#F8F5FF', ec='#a18cd1', lw=1.5, zorder=2)
    ax.add_patch(se)
    ax.text(9, 4.7, 'SEBlock 细节:  GlobalAvgPool → Linear(128→8) → ReLU → Linear(8→128) → Sigmoid'
            ' → 逐通道门控权重 ∈[0,1]   (瓶颈比 8:1 强制跨通道交互)',
            ha='center', fontsize=10, color='#555', zorder=3)

    arr(ax, 9, 4.1, 9, 3.65)

    # Conv 128→90
    box(ax, 9, 3.25, 13.0, 0.65, 'ConvBNRelu(128→90)   →   [B, 90, 14, 14]   (90 = 30 × 3, repeat=3)',
        '#FFE5E5', '#f5576c', fs=13)

    arr(ax, 9, 2.92, 9, 2.55)

    # Pool → Linear
    box(ax, 9, 2.15, 13.0, 0.65, 'AdaptiveAvgPool2d(1,1) → Squeeze → Linear(90→30)   3个独立预测器加权平均',
        '#FFD5D5', '#f5576c', fs=13)

    arr(ax, 9, 1.82, 9, 1.45)

    # 输出
    box(ax, 9, 1.05, 6.0, 0.6, 'pred_msg [B, 30]   预测消息比特',
        '#FFC0C0', '#f5576c', fs=14, fw='bold')

    fig_save(fig, 'diagram4_decoder.png')


if __name__ == '__main__':
    draw_overview()
    draw_featuregrid()
    draw_lowrank()
    draw_decoder()
    print('全部 4 张架构图已生成。')
