#!/usr/bin/env python3
"""生成水印大图，用于手动微信压缩测试。

用法:
  python generate_wechat_test.py --seed 0 --alpha 0.02
  python generate_wechat_test.py --seed 1 --alpha 0.03

输出: output1/wechat_manual_test/seed{N}/
  ├── full_4096x4096.png          # 完整水印大图 (发给微信)
  ├── full_4096x3072.png          # 裁剪版
  ├── cover_00.png ~ 03.png      # 4张底图
  ├── messages.json               # 消息记录
  └── crop_positions.json         # 用于解码的裁块位置
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class DotDict(dict):
    def __getattr__(self, key): return self.get(key)
    def __setattr__(self, key, value): self[key] = value


def load_model(ckpt_path: str, config_path: str):
    from model.ismark_v6_30bit import INRMark
    with open(config_path) as f:
        cfg = DotDict()
        for k, v in (yaml.safe_load(f) or {}).items():
            cfg[k] = DotDict(v) if isinstance(v, dict) else v

    model = INRMark(cfg)
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.cuda().eval(), cfg


def render_full_template(model, msg, cover_img_pil, size=2048, patch_size=256):
    """分块渲染 2048×2048 水印图，避免 OOM"""
    from dataset.Mydataset import generate_grid_coordinates
    import torch.nn.functional as F

    device = next(model.parameters()).device
    msg_tensor = torch.tensor([int(c) for c in msg], dtype=torch.float32).unsqueeze(0).to(device)

    # 底图 resize
    cover = cover_img_pil.resize((size, size), Image.BICUBIC)
    cover_arr = np.asarray(cover).astype(np.float32) / 255.0
    cover_tensor = torch.from_numpy(cover_arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # 全图坐标 [1, size, size, 2]，但在 GPU 上太大，分块处理
    coords_full = generate_grid_coordinates((-1, -1), 2, size).unsqueeze(0)  # CPU
    mask_full = torch.zeros(1, 3, size, size, device=device)
    output = torch.zeros(1, 3, size, size, device=device)

    stride = patch_size
    n_patches = 0
    with torch.no_grad():
        for y in range(0, size, stride):
            for x in range(0, size, stride):
                ye = min(y + stride, size)
                xe = min(x + stride, size)
                coords_patch = coords_full[:, y:ye, x:xe, :].to(device)
                cover_patch = cover_tensor[:, :, y:ye, x:xe]
                wm_patch, _ = model.render_img(coords_patch, msg_tensor, cover_patch)
                output[:, :, y:ye, x:xe] += wm_patch
                mask_full[:, :, y:ye, x:xe] += 1
                n_patches += 1

    output = output / mask_full.clamp_min(1)
    wm_arr = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    wm_arr = (wm_arr.clip(0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(wm_arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="消息种子 (0-9)")
    parser.add_argument("--alpha", type=float, default=0.02, help="水印强度")
    parser.add_argument("--display-test-dir", default="display_test2")
    args = parser.parse_args()

    # 加载消息
    manifest_path = Path(args.display_test_dir) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    msg_str = manifest[args.seed]["message"]
    print(f"[seed {args.seed}] message: {msg_str}")

    # 加载底图
    img_dir = Path("/mnt/xsj2023/Datasets/DIV2K/DIV2K_valid")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])

    cfg_path = "config/v6_size256_msg64.yaml"
    out_root = Path("output1/wechat_manual_test") / f"seed{args.seed:02d}"
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- 渲染函数（逐个加载模型避免 OOM） ----
    def render_with_ckpt(ckpt_path, name_tag):
        torch.cuda.empty_cache()
        model, _ = load_model(ckpt_path, cfg_path)
        model = model.to(device)
        wm_list = []
        for i in range(4):
            cover_pil = Image.open(img_paths[args.seed * 10 + i]).convert("RGB")
            wm = render_full_template(model, msg_str, cover_pil, 2048)
            wm_list.append(wm)
            # 只存一次 cover
            if name_tag == "v7":
                cover_pil.resize((2048, 2048), Image.BICUBIC).save(out_root / f"cover_{i:02d}.png")
        del model
        torch.cuda.empty_cache()
        return wm_list

    v7_ckpt = "output/ismark_v6_size256_msg64/lightning_logs/version_7/checkpoints/last.ckpt"
    wc_ckpt = "output1/wechat_test/ismark_v6_size256_msg64/lightning_logs/version_1/checkpoints/last.ckpt"

    print("Rendering v7 (original)...")
    full_wm_v7 = render_with_ckpt(v7_ckpt, "v7")
    print("Rendering wc (WeChat fine-tuned)...")
    full_wm_wc = render_with_ckpt(wc_ckpt, "wc")

    # ---- 拼成 4096×4096 ----
    def compose_4096(imgs):
        row1 = Image.new("RGB", (4096, 2048))
        row1.paste(imgs[0], (0, 0))
        row1.paste(imgs[1], (2048, 0))
        row2 = Image.new("RGB", (4096, 2048))
        row2.paste(imgs[2], (0, 0))
        row2.paste(imgs[3], (2048, 0))
        full = Image.new("RGB", (4096, 4096))
        full.paste(row1, (0, 0))
        full.paste(row2, (0, 2048))
        return full

    full_v7 = compose_4096(full_wm_v7)
    full_wc = compose_4096(full_wm_wc)

    # 裁成 4096×3072（模拟微信朋友圈裁剪）
    full_v7_3072 = full_v7.crop((0, 0, 4096, 3072))
    full_wc_3072 = full_wc.crop((0, 0, 4096, 3072))

    # 保存
    full_v7_3072.save(out_root / "v7_original_full_4096x3072.png")
    full_wc_3072.save(out_root / "wc_finetuned_full_4096x3072.png")
    full_v7.save(out_root / "v7_original_4096x4096.png")
    full_wc.save(out_root / "wc_finetuned_4096x4096.png")

    # 消息记录
    with open(out_root / "messages.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "message": msg_str,
            "msg_len": len(msg_str),
            "alpha": args.alpha,
        }, f, indent=2)

    # 裁块位置（16×12 = 192 个 256×256 块的位置）
    positions = []
    for row in range(12):
        for col in range(16):
            top = row * 256
            left = col * 256
            positions.append({"top": top, "left": left, "row": row, "col": col})
    with open(out_root / "crop_positions.json", "w") as f:
        json.dump(positions, f, indent=2)

    print(f"\nDone! Saved to: {out_root}")
    print(f"  发送以下图片给微信:")
    print(f"    v7:  {out_root / 'v7_original_full_4096x3072.png'}")
    print(f"    wc:  {out_root / 'wc_finetuned_full_4096x3072.png'}")
    print(f"\n  微信压缩后下载回来，保存到这个目录，然后告诉我解码。")


if __name__ == "__main__":
    main()
