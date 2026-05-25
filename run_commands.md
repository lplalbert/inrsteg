# 显示图片 & 检测水印 命令手册

## 目录结构

```
test_photo/v0_30_128/
├── full/
│   ├── sc/      (10 seeds: 0-9, 每 seed 30张)
│   └── scvx/    (10 seeds: 0-9, 每 seed 30张)
├── part/
│   ├── sc/      (10 seeds: 0-9, 每 seed 30张)
│   └── scvx/    (10 seeds: 0-9, 每 seed 30张)
└── 0/ ~ 9/      (原始 seed 目录)
```

---

## 一、显示水印图片 (display_for_camera.py)

```bash
# 显示某个 seed 的模板 (默认 alpha=1.0, 可调)
python display_for_camera.py --seed 0
python display_for_camera.py --seed 0 --alpha-scale 0.02
python display_for_camera.py --seed 0 --alpha-scale 0.015

# 换底图
python display_for_camera.py --seed 5 --image-index 3

# 预览模式 (不全屏)
python display_for_camera.py --seed 0 --no-fullscreen

# 批量生成 10 个模板 (种子 0-9)
python display_for_camera.py --gen-batch
```

---

## 二、检测水印 (decode_photo.py)

### 2.1 Full/sc (2000×1125)

```bash
# 不矫正 (预缩放 2000×1125, 避开边缘 QR 码)
python decode_photo.py \
  --photo-root /home/lpl2025/lpl/inrsteg-final_v1/test_photo/v0_30_128/001/full/sc \
  --pre-resize-w 2000 --pre-resize-h 1125 --edge-margin 100 \
  --gpu 2 --save-crops --extreme-top-k 50

# QR 矫正 global (全屏 1920×1080)
python decode_photo.py \
  --photo-root /home/lpl2025/lpl/inrsteg-final_v1/test_photo/v0_30_128/full/sc \
  --use-qr-correct --gpu 3 --save-crops --extreme-top-k 50

# QR 矫正 region (800×800)
python decode_photo.py \
  --photo-root /home/lpl2025/lpl/inrsteg-final_v1/test_photo/v0_30_128/full/sc \
  --use-qr-correct --region-size 800 --gpu 2 --save-crops --extreme-top-k 50
```

### 2.2 Full/scvx (2000×1125)

```bash
# 不矫正
python decode_photo.py \
  --photo-root /home/lpl2025/lpl/inrsteg-final_v1/test_photo/v0_30_128/full/scvx \
  --pre-resize-w 2000 --pre-resize-h 1125 --edge-margin 100 \
  --gpu 3 --save-crops --extreme-top-k 50

# QR 矫正 global
python decode_photo.py \
  --photo-root /home/lpl2025/lpl/inrsteg-final_v1/test_photo/v0_30_128/full/scvx \
  --use-qr-correct --gpu 2 --save-crops --extreme-top-k 50

# QR 矫正 region
python decode_photo.py \
  --photo-root /home/lpl2025/lpl/inrsteg-final_v1/test_photo/v0_30_128/full/scvx \
  --use-qr-correct --region-size 800 --gpu 2 --save-crops --extreme-top-k 50
```

### 2.3 Part/sc (1600×900)

```bash
# 不矫正 (预缩放 1600×900, 避开边缘 QR 码)
python decode_photo.py \
  --photo-root /home/lpl2025/lpl/inrsteg-final_v1/test_photo/v0_30_128/part/sc \
  --pre-resize-w 1600 --pre-resize-h 900 --edge-margin 100 \
  --gpu 3 --save-crops --extreme-top-k 50

# QR 矫正 global
python decode_photo.py \
  --photo-root /home/lpl2025/lpl/inrsteg-final_v1/test_photo/v0_30_128/part/sc \
  --use-qr-correct --gpu 2 --save-crops --extreme-top-k 50

# QR 矫正 region
python decode_photo.py \
  --photo-root /home/lpl2025/lpl/inrsteg-final_v1/test_photo/v0_30_128/part/sc \
  --use-qr-correct --region-size 800 --gpu 2 --save-crops --extreme-top-k 50
```

### 2.4 Part/scvx (1600×900)

```bash
# 不矫正
python decode_photo.py \
  --photo-root /home/lpl2025/lpl/inrsteg-final_v1/test_photo/v0_30_128/part/scvx \
  --pre-resize-w 1600 --pre-resize-h 900 --edge-margin 100 \
  --gpu 2 --save-crops --extreme-top-k 50

# QR 矫正 global
python decode_photo.py \
  --photo-root /home/lpl2025/lpl/inrsteg-final_v1/test_photo/v0_30_128/part/scvx \
  --use-qr-correct --gpu 2 --save-crops --extreme-top-k 50

# QR 矫正 region
python decode_photo.py \
  --photo-root /home/lpl2025/lpl/inrsteg-final_v1/test_photo/v0_30_128/part/scvx \
  --use-qr-correct --region-size 800 --gpu 2 --save-crops --extreme-top-k 50
```

---

## 三、参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--photo-root` | 照片根目录 (含 seed 子目录) | test_photo/v0_30_128 |
| `--use-qr-correct` | 启用 YOLO 二维码矫正后再裁剪 | 不启用 |
| `--region-size N` | 矫正后目标区域边长 (0=全屏) | 0 |
| `--pre-resize-w` | 不矫正时预缩放宽度 | 0 (不缩放) |
| `--pre-resize-h` | 不矫正时预缩放高度 | 0 (不缩放) |
| `--edge-margin` | 裁剪时避开边缘像素数 | 0 |
| `--gpu N` | GPU 编号 | 0 |
| `--save-crops` | 保存随机裁剪图 | 不保存 |
| `--extreme-top-k N` | 导出最高/最低 N 个裁剪 | 50 |
| `--debug-correct N` | 保存前 N 张矫正调试图 | 0 |
| `--crop-sizes` | 裁剪边长列表 | 128,256,512,1024 |
| `--num-crops-per-size` | 每个尺寸裁剪数 | 5 |
| `--qr-conf` | YOLO 置信度阈值 | 0.6 |
| `--screen-width` | 屏幕宽度 | 1920 |
| `--screen-height` | 屏幕高度 | 1080 |

---

## 四、结果输出

结果保存在 `photo-root` 目录下：

```
photo-root/
└── decode_photo_results/           # 不矫正
└── decode_photo_results_qr_correct/             # global 矫正
└── decode_photo_results_qr_correct_region800/   # region 矫正
    ├── crops.csv                   # 每个裁剪的详细信息
    ├── seed_summary.csv            # 每个 seed 汇总
    ├── image_summary.csv           # 每张图片汇总
    ├── crop_size_summary.csv       # 按裁剪边长汇总
    ├── crop_index_summary.csv      # 按裁剪序号汇总
    ├── crop_size_index_summary.csv # 按边长×序号汇总
    ├── summary.json                # 完整汇总 JSON
    ├── saved_crops/                # 保存的裁剪图 (--save-crops)
    ├── extreme_crops/              # 极值裁剪 (--extreme-top-k)
    └── debug_corrected/            # 矫正调试图 (--debug-correct)
```
