#!/usr/bin/env python3
"""解码微信压缩后的水印图，多尺度多模型对比，保存CSV。

用法:
  python decode_wechat_test.py
"""

import csv
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


def load_model(ckpt_path, cfg):
    from model.ismark_v6_30bit import INRMark
    m = INRMark(cfg)
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"], strict=False)
    return m.cuda().eval()


def main():
    with open("config/v6_size256_msg64.yaml") as f:
        cfg = DotDict()
        for k, v in (yaml.safe_load(f) or {}).items():
            cfg[k] = DotDict(v) if isinstance(v, dict) else v

    with open("output1/wechat_manual_test/batch/manifest.json") as f:
        manifest = json.load(f)
    msgs = [m["message"] for m in sorted(manifest, key=lambda x: x["index"])]

    base = Path("output1/wechat_manual_test/batch")
    crop_sizes = [256, 512, 1024]
    n_crops_per_size = {256: 20, 512: 10, 1024: 5}

    models = {
        "v7_original": "output/ismark_v6_size256_msg64/lightning_logs/version_7/checkpoints/last.ckpt",
        "wc_finetuned": "output1/wechat_test/ismark_v6_size256_msg64/lightning_logs/version_1/checkpoints/last.ckpt",
    }
    img_sets = ["v7_original_1", "wc_finetuned_1"]

    rows = []
    out_dir = Path("output1/wechat_test_decode")
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, ckpt in models.items():
        print(f"Loading {model_name}...")
        model = load_model(ckpt, cfg)

        for img_set in img_sets:
            img_dir = base / img_set
            files = sorted(img_dir.glob("*.jpg"))  # 按文件名 000~008 排序
            print(f"  {img_set}: {len(files)} images")

            for fi, p in enumerate(files):
                msg = msgs[fi] if fi < len(msgs) else msgs[0]
                msg_t = torch.tensor([int(c) for c in msg], dtype=torch.float32).cuda()
                img = Image.open(p).convert("RGB")
                H, W = img.size[1], img.size[0]

                for cs in crop_sizes:
                    if H < cs or W < cs:
                        continue
                    for ci in range(n_crops_per_size[cs]):
                        top = np.random.randint(0, H - cs)
                        left = np.random.randint(0, W - cs)
                        crop_pil = img.crop((left, top, left + cs, top + cs))
                        arr = np.asarray(crop_pil).astype(np.float32) / 255.0
                        crop_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
                        with torch.no_grad():
                            pred = model.decoder(crop_t)
                        acc = (1 - (torch.round(pred).clamp(0, 1) - msg_t).abs().sum() / 64).item()

                        rows.append({
                            "model": model_name,
                            "image_set": img_set,
                            "image_index": fi,
                            "image_file": p.name,
                            "crop_size": cs,
                            "crop_index": ci,
                            "top": top,
                            "left": left,
                            "bit_acc": acc,
                            "target_msg": msg,
                            "decoded_msg": "".join(str(int(b)) for b in torch.round(pred).clamp(0, 1).squeeze().cpu()[:64]),
                        })

        del model
        torch.cuda.empty_cache()

    # Write crops.csv
    with open(out_dir / "crops.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out_dir / 'crops.csv'}")

    # Summary by model + image_set + crop_size
    from collections import defaultdict
    summary = defaultdict(list)
    for r in rows:
        key = (r["model"], r["image_set"], r["crop_size"])
        summary[key].append(r["bit_acc"])

    print("\n=== Summary (mean / max / min) ===")
    summary_rows = []
    for (model, img_set, cs), accs in sorted(summary.items()):
        row = {
            "model": model, "image_set": img_set, "crop_size": cs,
            "count": len(accs), "mean": np.mean(accs), "max": np.max(accs),
            "min": np.min(accs), "std": np.std(accs),
        }
        summary_rows.append(row)
        print(f"{model:12s} | {img_set:16s} | c{cs:>4d} | mean={np.mean(accs):.4f} max={np.max(accs):.4f} min={np.min(accs):.4f}")

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved to {out_dir / 'summary.csv'}")

    # Per-image summary
    img_rows = []
    img_groups = defaultdict(list)
    for r in rows:
        img_groups[(r["model"], r["image_set"], r["image_index"])].append(r["bit_acc"])
    for (model, img_set, idx), accs in sorted(img_groups.items()):
        img_rows.append({"model": model, "image_set": img_set, "image_index": idx, "mean": np.mean(accs), "max": np.max(accs), "min": np.min(accs)})
    with open(out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=img_rows[0].keys())
        w.writeheader()
        w.writerows(img_rows)

    print(f"\nAll results saved to {out_dir}/")


if __name__ == "__main__":
    main()
