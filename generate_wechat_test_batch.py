#!/usr/bin/env python3
"""批量生成水印图，用于微信压缩测试。

每张图：单张底图放大到4096×4096 → 2048水印贴2×2 → 裁4096×3072
两个模型版本各30张，共60张。

输出: output1/wechat_manual_test/batch/
  ├── v7_original/img_000.png ~ 029.png    # 原始模型水印图
  ├── wc_finetuned/img_000.png ~ 029.png   # 微调模型水印图
  ├── covers/img_000_cover.png             # 底图
  └── manifest.json                        # 消息记录
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


def load_model(ckpt_path: str, config_path: str, device):
    from model.ismark_v6_30bit import INRMark
    with open(config_path) as f:
        cfg = DotDict()
        for k, v in (yaml.safe_load(f) or {}).items():
            cfg[k] = DotDict(v) if isinstance(v, dict) else v

    model = INRMark(cfg)
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), cfg


def render_one(model, msg_str, cover_pil, device, wm_size=2048, canvas_size=4096, patch_size=512):
    """分块渲染水印大图，避免 OOM。"""
    from dataset.Mydataset import generate_grid_coordinates

    msg = torch.tensor([int(c) for c in msg_str], dtype=torch.float32).unsqueeze(0).to(device)

    # 底图放大到 canvas_size
    cover = cover_pil.convert("RGB").resize((canvas_size, canvas_size), Image.BICUBIC)
    cover_arr = np.asarray(cover).astype(np.float32) / 255.0
    cover_tensor = torch.from_numpy(cover_arr).permute(2, 0, 1).unsqueeze(0)

    # 完整 4096×4096 的坐标（CPU），逐块处理
    coords_full = generate_grid_coordinates((-1, -1), 2, canvas_size).unsqueeze(0)  # [1, 4096, 4096, 2] CPU

    output = np.zeros((canvas_size, canvas_size, 3), dtype=np.float32)
    mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)

    n_patches = 0
    with torch.no_grad():
        for y in range(0, canvas_size, patch_size):
            for x in range(0, canvas_size, patch_size):
                ye = min(y + patch_size, canvas_size)
                xe = min(x + patch_size, canvas_size)
                hh, ww = ye - y, xe - x

                coords_p = coords_full[:, y:ye, x:xe, :].to(device)
                cover_p = cover_tensor[:, :, y:ye, x:xe].to(device)
                wm_p, _ = model.render_img(coords_p, msg, cover_p)
                wm_np = wm_p.squeeze(0).permute(1, 2, 0).cpu().numpy()
                output[y:ye, x:xe] += wm_np
                mask[y:ye, x:xe] += 1
                n_patches += 1

    output = output / mask[:, :, None]
    output = (output.clip(0, 1) * 255).round().astype(np.uint8)
    result = Image.fromarray(output)

    # 裁 4096×3072
    return result.crop((0, 0, 4096, 3072))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于选底图和生成消息）")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    cfg_path = "config/v6_size256_msg64.yaml"

    out_root = Path("output1/wechat_manual_test/batch")
    (out_root / "v7_original").mkdir(parents=True, exist_ok=True)
    (out_root / "wc_finetuned").mkdir(parents=True, exist_ok=True)
    (out_root / "covers").mkdir(parents=True, exist_ok=True)

    # 加载底图列表
    img_dir = Path("/home/lpl2025/lpl/inrsteg-final_v1/raw_input")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
    args.num_images = min(args.num_images, len(img_paths))
    covers = [Image.open(p).convert("RGB") for p in img_paths[:args.num_images]]

    # 生成30个随机64-bit消息
    msgs = []
    for i in range(args.num_images):
        bits = "".join(str(b) for b in np.random.randint(0, 2, 64))
        msgs.append(bits)

    # 加载 v7 模型
    print("Loading v7 (original)...")
    v7_ckpt = "output/ismark_v6_size256_msg64/lightning_logs/version_7/checkpoints/last.ckpt"
    model_v7, _ = load_model(v7_ckpt, cfg_path, device)

    # 渲染 v7
    print(f"Rendering {args.num_images} images with v7...")
    manifest = []
    for i in range(args.num_images):
        wm_img = render_one(model_v7, msgs[i], covers[i], device)
        wm_img.save(out_root / "v7_original" / f"img_{i:03d}.png")
        covers[i].resize((4096, 4096), Image.BICUBIC).save(out_root / "covers" / f"img_{i:03d}_cover.png")
        print(f"  v7 [{i+1}/{args.num_images}]")
        manifest.append({"index": i, "message": msgs[i], "cover": img_paths[i].name})

    del model_v7
    torch.cuda.empty_cache()

    # 加载 wc 模型
    print("Loading wc (WeChat fine-tuned)...")
    wc_ckpt = "output1/wechat_test/ismark_v6_size256_msg64/lightning_logs/version_1/checkpoints/last.ckpt"
    model_wc, _ = load_model(wc_ckpt, cfg_path, device)

    # 渲染 wc
    print(f"Rendering {args.num_images} images with wc...")
    for i in range(args.num_images):
        wm_img = render_one(model_wc, msgs[i], covers[i], device)
        wm_img.save(out_root / "wc_finetuned" / f"img_{i:03d}.png")
        print(f"  wc [{i+1}/{args.num_images}]")

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone! {args.num_images*2} images saved to {out_root}")
    print(f"  v7_original:  {out_root / 'v7_original'}/*.png")
    print(f"  wc_finetuned: {out_root / 'wc_finetuned'}/*.png")
    print(f"  manifest:      {out_root / 'manifest.json'}")
    print(f"\n将两目录各30张图通过微信发送→下载回来→告诉我解码")


if __name__ == "__main__":
    main()
