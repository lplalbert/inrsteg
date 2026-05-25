"""
解码手机拍的照片中的水印

用法:
  python decode_photo.py --photo 照片.jpg
  python decode_photo.py --photo-root /home/lpl2025/lpl/inrsteg-final_v1/test_photo/v0_30_128
"""

import argparse
import csv
import importlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from ultralytics import YOLO

    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DotDict(dict):
    def __getattr__(self, key):
        return self.get(key, None)

    def __setattr__(self, key, value):
        self[key] = value


def load_yaml_args(path: str) -> DotDict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    args = DotDict()
    for key, value in data.items():
        args[key] = DotDict(value) if isinstance(value, dict) else value
    return args


def pil_to_tensor(img: Image.Image, device: str = None) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).clamp(0, 1)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().float()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.clamp(0, 1)
    arr = (tensor[:3].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def message_to_tensor(msg: str, msg_len: int) -> torch.Tensor:
    if len(msg) != msg_len:
        raise ValueError(f"message must have exactly {msg_len} bits, got {len(msg)}")
    if any(ch not in "01" for ch in msg):
        raise ValueError("message must be a binary string")
    return torch.tensor([int(ch) for ch in msg], dtype=torch.float32)


def tensor_to_bits(tensor: torch.Tensor, msg_len: int) -> str:
    bits = torch.round(tensor).clamp(0, 1).squeeze(0).detach().cpu()
    return "".join(str(int(x)) for x in bits[:msg_len])


def msg_acc(predict_msg: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
    decoded = torch.round(predict_msg).clamp(0, 1).detach()
    bit_err = torch.abs(decoded - msg).sum() / msg.numel()
    return 1 - bit_err


def decode_bits(model, img: torch.Tensor, decode_size: int, msg_len: int) -> Tuple[str, torch.Tensor]:
    if img.shape[-2:] != (decode_size, decode_size):
        img = F.interpolate(img, size=(decode_size, decode_size), mode="bilinear", align_corners=False)
    pred = model.decode_msg(img)
    return tensor_to_bits(pred, msg_len), pred


def decode_crops_batch(model, crops: torch.Tensor, decode_size: int, msg_len: int) -> Tuple[List[str], torch.Tensor]:
    """批量解码。crops: (B, 3, H, W) → preds: (B, msg_len)"""
    if crops.shape[-2:] != (decode_size, decode_size):
        crops = F.interpolate(crops, size=(decode_size, decode_size), mode="bilinear", align_corners=False)
    preds = model.decode_msg(crops)  # (B, msg_len)
    decoded_msgs = []
    for i in range(preds.shape[0]):
        bits = torch.round(preds[i]).clamp(0, 1).detach().cpu()
        decoded_msgs.append("".join(str(int(x)) for x in bits[:msg_len]))
    return decoded_msgs, preds


def msg_acc_batch(preds: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
    """批量计算准确率。preds: (B, msg_len), msg: (1, msg_len) → (B,)"""
    decoded = torch.round(preds).clamp(0, 1).detach()
    bit_err = (decoded - msg).abs().sum(dim=1) / msg.shape[1]
    return 1 - bit_err


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def variance(values: List[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return float(sum((value - avg) ** 2 for value in values) / len(values))


def metric_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "var": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    var = variance(values)
    return {
        "count": len(values),
        "mean": mean(values),
        "var": var,
        "std": float(var ** 0.5),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def resolve_ckpt(raw: str) -> str:
    if raw and os.path.exists(raw):
        return raw
    if raw and "/output/" in raw:
        local = "./output/" + raw.split("/output/", 1)[1]
        if os.path.exists(local):
            return local
    raise FileNotFoundError(f"Checkpoint not found: {raw}")


def load_seed_messages(display_test_dir: str) -> Dict[str, str]:
    base = Path(display_test_dir)
    messages: Dict[str, str] = {}
    manifest_path = base / "manifest.json"

    if manifest_path.exists():
        with manifest_path.open("r") as f:
            manifest = json.load(f)
        for item in manifest:
            seed = str(item["seed"])
            message = str(item["message"]).strip()
            if message:
                messages[seed] = message

    for path in sorted(base.glob("message_seed*.txt")):
        seed = path.stem.replace("message_seed", "")
        message = path.read_text().strip()
        if message and seed not in messages:
            messages[seed] = message

    return messages


def list_seed_dirs(photo_root: str) -> List[Path]:
    root = Path(photo_root)
    if not root.exists():
        raise FileNotFoundError(f"Photo root not found: {photo_root}")
    return [path for path in sorted(root.iterdir(), key=lambda p: p.name) if path.is_dir()]


def list_images(img_dir: Path) -> List[Path]:
    paths = [path for path in sorted(img_dir.iterdir(), key=lambda p: p.name) if path.suffix.lower() in IMAGE_EXTS]
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return paths


def parse_crop_sizes(raw: str) -> List[int]:
    sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("crop sizes must be positive integers")
    return sizes


def random_crop_tensor(img: torch.Tensor, crop_size: int, edge_margin: int = 0) -> Tuple[torch.Tensor, Dict[str, int]]:
    _, _, h, w = img.shape
    if h < crop_size or w < crop_size:
        crop = F.interpolate(img, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        return crop, {"top": 0, "left": 0, "size": crop_size, "resized": 1, "source_h": h, "source_w": w}

    margin = min(edge_margin, (h - crop_size) // 2, (w - crop_size) // 2)
    top = random.randint(margin, h - crop_size - margin)
    left = random.randint(margin, w - crop_size - margin)
    crop = img[:, :, top : top + crop_size, left : left + crop_size]
    return crop, {"top": top, "left": left, "size": crop_size, "resized": 0, "source_h": h, "source_w": w}


def save_crop_image(path: Path, crop_tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(crop_tensor).save(path)


def order_points_tl_tr_br_bl(points: List[List[float]]) -> List[List[float]]:
    """将 4 个点按 TL, TR, BR, BL 顺序排列。"""
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return [
        [float(tl[0]), float(tl[1])],
        [float(tr[0]), float(tr[1])],
        [float(br[0]), float(br[1])],
        [float(bl[0]), float(bl[1])],
    ]


def detect_qr_markers_yolo(model_yolo, img_cv: np.ndarray, conf: float = 0.6, device: str = "cuda:0") -> List[Dict]:
    """用 YOLO 检测 4 个 QR 标记点。返回按 TL,TR,BR,BL 排序的标记列表。"""
    results = model_yolo(img_cv, verbose=False, conf=conf, device=device)
    if not results:
        return []

    result = results[0]
    if result.boxes is None or result.boxes.xywh is None or len(result.boxes.xywh) < 4:
        return []

    scored = []
    for xywh, xyxy, score in zip(result.boxes.xywh, result.boxes.xyxy, result.boxes.conf):
        cx, cy, _, _ = xywh.tolist()
        x1, y1, x2, y2 = xyxy.tolist()
        scored.append(
            {
                "center": [float(cx), float(cy)],
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    top4 = scored[:4]

    ordered_centers = order_points_tl_tr_br_bl([m["center"] for m in top4])

    ordered = []
    used = [False] * len(top4)
    for p in ordered_centers:
        best_idx = -1
        best_dist = 1e18
        for i, m in enumerate(top4):
            if used[i]:
                continue
            dx = m["center"][0] - p[0]
            dy = m["center"][1] - p[1]
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        used[best_idx] = True
        ordered.append(top4[best_idx])

    return ordered


def correct_photo_by_qr(
    img_pil: Image.Image,
    yolo_model,
    screen_w: int = 1920,
    screen_h: int = 1080,
    qr_size: int = 120,
    conf: float = 0.6,
    region_size: int = 0,
    device: str = "cuda:0",
) -> Tuple[Optional[Image.Image], Dict]:
    """用 YOLO 检测 4 个 QR 标记点，做透视矫正。

    region_size=0: global 模式，QR→屏幕四角 (screen_w × screen_h)
    region_size>0: region 模式，QR→区域四角 (region_size × region_size)，仅保留区域内内容
    """
    if not HAS_CV2 or not HAS_YOLO:
        raise ImportError("QR correction requires opencv-python-headless and ultralytics.")

    img_cv = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    markers = detect_qr_markers_yolo(yolo_model, img_cv, conf=conf, device=device)

    info = {
        "qr_detected": False,
        "num_markers": len(markers),
        "markers": None,
        "src_points": None,
        "dst_points": None,
        "region_size": region_size,
    }

    if len(markers) != 4:
        return None, info

    info["markers"] = [
        {"center": m["center"], "box": m["box"], "score": m["score"]} for m in markers
    ]

    # 源点：4 个 marker 的中心，按 TL,TR,BR,BL 排序
    src_points = order_points_tl_tr_br_bl([m["center"] for m in markers])
    info["src_points"] = src_points

    if region_size > 0:
        # region 模式：QR 中心直接映射到区域四角
        dst_points = [
            [0.0, 0.0],
            [float(region_size), 0.0],
            [float(region_size), float(region_size)],
            [0.0, float(region_size)],
        ]
        warp_w, warp_h = region_size, region_size
    else:
        # global 模式：QR 中心映射到屏幕四角（qr_size 偏移）
        half_qr = qr_size / 2.0
        dst_points = [
            [half_qr, half_qr],
            [screen_w - half_qr, half_qr],
            [screen_w - half_qr, screen_h - half_qr],
            [half_qr, screen_h - half_qr],
        ]
        warp_w, warp_h = screen_w, screen_h

    info["dst_points"] = dst_points
    info["warp_w"] = warp_w
    info["warp_h"] = warp_h

    M = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))
    corrected_cv = cv2.warpPerspective(img_cv, M, (warp_w, warp_h), flags=cv2.INTER_CUBIC)
    corrected_rgb = cv2.cvtColor(corrected_cv, cv2.COLOR_BGR2RGB)
    corrected_pil = Image.fromarray(corrected_rgb)

    return corrected_pil, info


def draw_correction_debug(
    original_pil: Image.Image,
    corrected_pil: Optional[Image.Image],
    info: Dict,
    save_path: Path,
) -> None:
    """绘制矫正调试图：原图（标注 4 个 QR 标记） + 矫正后的图。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    original = original_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(original)

    # 绘制 YOLO 检测到的标记框和中心
    if info.get("markers"):
        for i, m in enumerate(info["markers"]):
            x1, y1, x2, y2 = [int(v) for v in m["box"]]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=3)
            cx, cy = int(m["center"][0]), int(m["center"][1])
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 0, 0))
            draw.text((cx + 6, cy - 10), f"M{i}", fill=(255, 0, 0))

    # 绘制用于透视变换的源点连线
    if info.get("src_points"):
        pts = [(p[0], p[1]) for p in info["src_points"]]
        for i in range(4):
            j = (i + 1) % 4
            draw.line([pts[i], pts[j]], fill=(0, 255, 0), width=3)
        for i, p in enumerate(pts):
            draw.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=(0, 255, 0))
            draw.text((p[0] + 8, p[1] + 8), f"S{i}", fill=(0, 255, 0))

    corrected = (
        corrected_pil.copy()
        if corrected_pil
        else Image.new("RGB", (original.width // 4, original.height // 4), (128, 128, 128))
    )

    orig_w, orig_h = original.size
    corr_w, corr_h = corrected.size
    max_w = max(orig_w, corr_w)
    debug_img = Image.new("RGB", (max_w, orig_h + corr_h), (64, 64, 64))
    debug_img.paste(original, (0, 0))
    debug_img.paste(corrected.resize((corr_w, corr_h)), (0, orig_h))
    debug_img.save(save_path)


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_stats_row(prefix: str, key_value: Dict[str, object], values: List[float]) -> Dict[str, object]:
    stats = metric_stats(values)
    row = dict(key_value)
    row[f"{prefix}_count"] = stats["count"]
    row[f"{prefix}_mean"] = stats["mean"]
    row[f"{prefix}_var"] = stats["var"]
    row[f"{prefix}_std"] = stats["std"]
    row[f"{prefix}_min"] = stats["min"]
    row[f"{prefix}_max"] = stats["max"]
    return row


def export_extreme_crops(crop_rows: List[Dict], result_dir: Path, top_k: int) -> None:
    if not crop_rows or top_k <= 0:
        return

    extreme_dir = result_dir / "extreme_crops"
    low_dir = extreme_dir / "lowest"
    high_dir = extreme_dir / "highest"
    low_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)

    sortable_rows = [row for row in crop_rows if row.get("saved_crop_path")]
    if not sortable_rows:
        return

    sorted_rows = sorted(
        sortable_rows,
        key=lambda row: (float(row["bit_acc"]), row["seed"], row["image_name"], row["crop_size"], row["crop_index"]),
    )
    lowest_rows = sorted_rows[: min(top_k, len(sorted_rows))]
    highest_rows = list(reversed(sorted_rows[-min(top_k, len(sorted_rows)) :]))

    def copy_rows(rows: List[Dict], target_dir: Path) -> List[Dict]:
        exported = []
        for rank, row in enumerate(rows, start=1):
            source_path = row["saved_crop_path"]
            out_name = (
                f"rank{rank:03d}_acc{float(row['bit_acc']):.4f}_seed{row['seed']}_"
                f"{Path(source_path).name}"
            )
            target_path = target_dir / out_name
            shutil.copy2(source_path, target_path)
            exported.append(
                {
                    "rank": rank,
                    "bit_acc": row["bit_acc"],
                    "seed": row["seed"],
                    "image_name": row["image_name"],
                    "crop_size": row["crop_size"],
                    "crop_index": row["crop_index"],
                    "top": row["top"],
                    "left": row["left"],
                    "decoded_msg": row["decoded_msg"],
                    "target_msg": row["target_msg"],
                    "saved_crop_path": str(target_path),
                }
            )
        return exported

    write_csv(extreme_dir / "lowest.csv", copy_rows(lowest_rows, low_dir))
    write_csv(extreme_dir / "highest.csv", copy_rows(highest_rows, high_dir))


def decode_single_photo(cli, model, msg_len: int, decode_size: int) -> None:
    img = Image.open(cli.photo).convert("RGB")
    tensor = pil_to_tensor(img, cli.device)
    print(f"[decode] 原始照片: {img.size}")

    with torch.no_grad():
        decoded_msg, pred = decode_bits(model, tensor, decode_size, msg_len)

    print(f"\n{'=' * 60}")
    print(f"解码结果 ({msg_len}bit):")
    print(f"  {decoded_msg}")
    if cli.target_msg:
        target = cli.target_msg.strip()
        target_tensor = message_to_tensor(target, msg_len).unsqueeze(0).to(cli.device)
        acc = float(msg_acc(pred, target_tensor).item())
        match = int(round(acc * msg_len))
        print(f"\n  目标消息: {target}")
        print(f"  比特准确率: {match}/{msg_len} = {acc:.4f} ({acc * 100:.1f}%)")
        if acc > 0.9:
            print("  结论: 水印提取成功 ✓")
        elif acc > 0.7:
            print("  结论: 部分提取, 可能有戏 △")
        else:
            print("  结论: 提取失败, 照片质量可能不够 ✗")
    print(f"{'=' * 60}")


def decode_batch(cli, model, msg_len: int, decode_size: int) -> None:
    seed_messages = load_seed_messages(cli.display_test_dir)
    if not seed_messages:
        raise FileNotFoundError(f"No seed messages found in {cli.display_test_dir}")

    crop_sizes = parse_crop_sizes(cli.crop_sizes)
    result_dir = Path(cli.result_dir)
    crop_output_dir = result_dir / "saved_crops"
    seed_dirs = list_seed_dirs(cli.photo_root)

    crop_rows: List[Dict] = []
    seed_summary_rows: List[Dict] = []
    image_summary_rows: List[Dict] = []
    size_scores: Dict[int, List[float]] = {size: [] for size in crop_sizes}
    index_scores: Dict[int, List[float]] = {idx: [] for idx in range(cli.num_crops_per_size)}
    size_index_scores: Dict[Tuple[int, int], List[float]] = {
        (size, idx): [] for size in crop_sizes for idx in range(cli.num_crops_per_size)
    }
    total_image_count = 0
    debug_correct_dir = result_dir / "debug_corrected"
    debug_correct_count = 0
    qr_correct_total = 0
    qr_correct_success = 0

    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        target_msg = seed_messages.get(seed_name)
        if target_msg is None:
            print(f"[warn] seed {seed_name} has no message in {cli.display_test_dir}, skipped")
            continue

        target_msg_tensor = message_to_tensor(target_msg, msg_len).unsqueeze(0).to(cli.device)
        image_paths = list_images(seed_dir)
        total_image_count += len(image_paths)
        seed_scores: List[float] = []
        seed_size_scores: Dict[int, List[float]] = {size: [] for size in crop_sizes}

        print(f"[seed {seed_name}] images={len(image_paths)} target={target_msg}")

        for image_index, image_path in enumerate(image_paths):
            img = Image.open(image_path).convert("RGB")
            qr_corrected = False
            qr_info: Dict = {}
            image_scores: List[float] = []
            image_size_scores: Dict[int, List[float]] = {size: [] for size in crop_sizes}

            if cli.no_crop:
                # 直接全图解码，不做裁剪和矫正
                img_tensor = pil_to_tensor(img).to(cli.device)
                with torch.no_grad():
                    decoded_msg, pred = decode_bits(model, img_tensor, decode_size, msg_len)
                acc_val = float(msg_acc(pred, target_msg_tensor).item())

                seed_scores.append(acc_val)
                image_scores.append(acc_val)
                for size in crop_sizes:
                    seed_size_scores[size].append(acc_val)
                    image_size_scores[size].append(acc_val)

                crop_rows.append({
                    "seed": seed_name, "image_path": str(image_path),
                    "image_name": image_path.name, "image_index": image_index,
                    "qr_corrected": 0, "crop_size": 0, "crop_index": 0,
                    "top": 0, "left": 0,
                    "source_h": img.size[1], "source_w": img.size[0],
                    "resized": 1, "decoded_msg": decoded_msg,
                    "target_msg": target_msg, "bit_acc": acc_val,
                    "saved_crop_path": "",
                })

                image_stats = metric_stats(image_scores)
                image_row = {
                    "seed": seed_name, "image_name": image_path.name,
                    "image_path": str(image_path), "image_index": image_index,
                    "qr_corrected": 0, "crop_count": 1,
                    "bit_acc_mean": image_stats["mean"],
                    "bit_acc_var": image_stats["var"],
                    "bit_acc_std": image_stats["std"],
                    "bit_acc_min": image_stats["min"],
                    "bit_acc_max": image_stats["max"],
                }
                for crop_size in crop_sizes:
                    image_row[f"bit_acc_{crop_size}_mean"] = acc_val
                image_summary_rows.append(image_row)
                continue

            if cli.use_qr_correct:
                qr_correct_total += 1
                debug = cli.debug_correct > 0 and debug_correct_count < cli.debug_correct
                corrected, qr_info = correct_photo_by_qr(
                    img,
                    yolo_model=cli.qr_yolo_model,
                    screen_w=cli.screen_width,
                    screen_h=cli.screen_height,
                    qr_size=cli.qr_size,
                    conf=cli.qr_conf,
                    region_size=cli.region_size,
                    device=cli.device,
                )
                if corrected is not None:
                    qr_corrected = True
                    qr_correct_success += 1
                    img_processed = corrected
                    if debug:
                        draw_correction_debug(img, corrected, qr_info, debug_correct_dir / f"{seed_name}_{image_path.stem}_debug.png")
                        debug_correct_count += 1
                else:
                    img_processed = img
            else:
                img_processed = img

            # 预缩放：相机像素远高于屏幕分辨率，先 resize 到固定大小
            if cli.pre_resize_w > 0 and cli.pre_resize_h > 0:
                img_processed = img_processed.resize((cli.pre_resize_w, cli.pre_resize_h), Image.BICUBIC)

            img_tensor = pil_to_tensor(img_processed)

            for crop_size in crop_sizes:
                # 收集同尺寸的全部裁剪，打包成 batch 一次解码
                batch_crops = []
                batch_metas = []
                for crop_index in range(cli.num_crops_per_size):
                    crop_tensor, crop_meta = random_crop_tensor(img_tensor, crop_size, edge_margin=cli.edge_margin)
                    batch_crops.append(crop_tensor)
                    batch_metas.append((crop_index, crop_meta))

                batch_tensor = torch.cat(batch_crops, dim=0).to(cli.device)
                with torch.no_grad():
                    decoded_msgs, preds = decode_crops_batch(model, batch_tensor, decode_size, msg_len)
                accs = msg_acc_batch(preds, target_msg_tensor)

                for (crop_index, crop_meta), decoded_msg, acc in zip(batch_metas, decoded_msgs, accs):
                    acc_val = float(acc.item())

                    if cli.save_crops:
                        crop_filename = (
                            f"{image_path.stem}_img{image_index:03d}_crop{crop_size}_idx{crop_index:02d}.png"
                        )
                        saved_crop_path = crop_output_dir / seed_name / crop_filename
                        # 保存原始裁剪（未 resize 到 decode_size 的版本）
                        save_crop_image(saved_crop_path, batch_crops[crop_index])
                        saved_crop_path_str = str(saved_crop_path)
                    else:
                        saved_crop_path_str = ""

                    seed_scores.append(acc_val)
                    seed_size_scores[crop_size].append(acc_val)
                    image_scores.append(acc_val)
                    image_size_scores[crop_size].append(acc_val)
                    size_scores[crop_size].append(acc_val)
                    index_scores[crop_index].append(acc_val)
                    size_index_scores[(crop_size, crop_index)].append(acc_val)

                    crop_rows.append(
                        {
                            "seed": seed_name,
                            "image_path": str(image_path),
                            "image_name": image_path.name,
                            "image_index": image_index,
                            "qr_corrected": int(qr_corrected),
                            "crop_size": crop_meta["size"],
                            "crop_index": crop_index,
                            "top": crop_meta["top"],
                            "left": crop_meta["left"],
                            "source_h": crop_meta["source_h"],
                            "source_w": crop_meta["source_w"],
                            "resized": crop_meta["resized"],
                            "decoded_msg": decoded_msg,
                            "target_msg": target_msg,
                            "bit_acc": acc_val,
                            "saved_crop_path": saved_crop_path_str,
                        }
                    )

            image_stats = metric_stats(image_scores)
            image_row = {
                "seed": seed_name,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "image_index": image_index,
                "qr_corrected": int(qr_corrected),
                "crop_count": len(image_scores),
                "bit_acc_mean": image_stats["mean"],
                "bit_acc_var": image_stats["var"],
                "bit_acc_std": image_stats["std"],
                "bit_acc_min": image_stats["min"],
                "bit_acc_max": image_stats["max"],
            }
            for crop_size in crop_sizes:
                image_row[f"bit_acc_{crop_size}_mean"] = metric_stats(image_size_scores[crop_size])["mean"]
            image_summary_rows.append(image_row)

        seed_stats = metric_stats(seed_scores)
        seed_row = {
            "seed": seed_name,
            "image_count": len(image_paths),
            "crop_count": len(seed_scores),
            "bit_acc_mean": seed_stats["mean"],
            "bit_acc_var": seed_stats["var"],
            "bit_acc_std": seed_stats["std"],
            "bit_acc_min": seed_stats["min"],
            "bit_acc_max": seed_stats["max"],
        }
        for crop_size in crop_sizes:
            seed_row[f"bit_acc_{crop_size}_mean"] = metric_stats(seed_size_scores[crop_size])["mean"]
        seed_summary_rows.append(seed_row)

        size_text = "  ".join(
            f"mean@{crop_size}={metric_stats(seed_size_scores[crop_size])['mean']:.4f}" for crop_size in crop_sizes
        )
        print(
            f"  -> crops={len(seed_scores)}  mean={seed_stats['mean']:.4f}  "
            f"min={seed_stats['min']:.4f}  max={seed_stats['max']:.4f}  {size_text}"
        )

    crop_size_summary_rows = [
        build_stats_row("bit_acc", {"crop_size": crop_size}, size_scores[crop_size]) for crop_size in crop_sizes
    ]
    crop_index_summary_rows = [
        build_stats_row("bit_acc", {"crop_index": crop_index}, index_scores[crop_index])
        for crop_index in range(cli.num_crops_per_size)
    ]
    crop_size_index_summary_rows = [
        build_stats_row(
            "bit_acc",
            {"crop_size": crop_size, "crop_index": crop_index},
            size_index_scores[(crop_size, crop_index)],
        )
        for crop_size in crop_sizes
        for crop_index in range(cli.num_crops_per_size)
    ]

    write_csv(result_dir / "crops.csv", crop_rows)
    write_csv(result_dir / "seed_summary.csv", seed_summary_rows)
    write_csv(result_dir / "image_summary.csv", image_summary_rows)
    write_csv(result_dir / "crop_size_summary.csv", crop_size_summary_rows)
    write_csv(result_dir / "crop_index_summary.csv", crop_index_summary_rows)
    write_csv(result_dir / "crop_size_index_summary.csv", crop_size_index_summary_rows)
    export_extreme_crops(crop_rows, result_dir, cli.extreme_top_k)

    overall_scores = [row["bit_acc"] for row in crop_rows]
    summary = {
        "cfg": cli.cfg,
        "ckpt": resolve_ckpt(cli.ckpt),
        "model_module": cli.model_module,
        "photo_root": cli.photo_root,
        "display_test_dir": cli.display_test_dir,
        "random_seed": cli.random_seed,
        "decode_size": decode_size,
        "msg_len": msg_len,
        "gpu": cli.gpu,
        "crop_sizes": crop_sizes,
        "num_crops_per_size": cli.num_crops_per_size,
        "use_qr_correct": cli.use_qr_correct,
        "region_size": cli.region_size,
        "qr_correct_total": qr_correct_total,
        "qr_correct_success": qr_correct_success,
        "num_seeds": len(seed_summary_rows),
        "num_images": total_image_count,
        "num_crops": len(crop_rows),
        "overall": metric_stats(overall_scores),
        "by_crop_size": {str(crop_size): metric_stats(size_scores[crop_size]) for crop_size in crop_sizes},
        "by_crop_index": {str(crop_index): metric_stats(index_scores[crop_index]) for crop_index in range(cli.num_crops_per_size)},
        "by_crop_size_and_index": {
            f"{crop_size}_{crop_index}": metric_stats(size_index_scores[(crop_size, crop_index)])
            for crop_size in crop_sizes
            for crop_index in range(cli.num_crops_per_size)
        },
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 72}")
    if cli.use_qr_correct:
        print(f"二维码矫正: {qr_correct_success}/{qr_correct_total} 张图片检测到二维码并完成矫正")
    if cli.gpu is not None:
        print(f"GPU 设备: cuda:{cli.gpu}")
    print("每个 seed 的准确率汇总:")
    for row in seed_summary_rows:
        print(
            f"seed={row['seed']:>2}  images={int(row['image_count']):>3}  crops={int(row['crop_count']):>4}  "
            f"mean={float(row['bit_acc_mean']):.4f}  min={float(row['bit_acc_min']):.4f}  "
            f"max={float(row['bit_acc_max']):.4f}"
        )
    print("\n每张图片准确率:")
    for row in image_summary_rows[:10]:
        print(
            f"seed={row['seed']:>2}  img={row['image_name'][:30]:<30}  crops={int(row['crop_count']):>4}  "
            f"mean={float(row['bit_acc_mean']):.4f}  min={float(row['bit_acc_min']):.4f}  "
            f"max={float(row['bit_acc_max']):.4f}"
        )
    if len(image_summary_rows) > 10:
        print(f"  ... 还有 {len(image_summary_rows) - 10} 张图片")
    print("\n不同边长准确率:")
    for row in crop_size_summary_rows:
        print(
            f"size={int(row['crop_size']):>4}  count={int(row['bit_acc_count']):>5}  "
            f"mean={float(row['bit_acc_mean']):.4f}  min={float(row['bit_acc_min']):.4f}  "
            f"max={float(row['bit_acc_max']):.4f}"
        )
    print("\n不同序号准确率:")
    for row in crop_index_summary_rows:
        print(
            f"index={int(row['crop_index']):>2}  count={int(row['bit_acc_count']):>5}  "
            f"mean={float(row['bit_acc_mean']):.4f}  min={float(row['bit_acc_min']):.4f}  "
            f"max={float(row['bit_acc_max']):.4f}"
        )
    if cli.save_crops:
        print(f"裁剪图已保存到: {crop_output_dir}")
    if cli.extreme_top_k > 0:
        print(f"极值裁剪已导出到: {result_dir / 'extreme_crops'}")
    print(f"结果已保存到: {result_dir}")
    print(f"{'=' * 72}")


def main():
    parser = argparse.ArgumentParser(description="解码手机照片中的水印")
    parser.add_argument("--photo", default=None, help="单张手机拍摄照片路径")
    parser.add_argument("--photo-root", default="test_photo/v1_64_256")
    parser.add_argument("--display-test-dir", default="display_test2")
    parser.add_argument("--cfg", default="/home/lpl2025/lpl/inrsteg-final_v1/config/v6_size256_msg64.yaml")
    parser.add_argument(
        "--ckpt",
        default="/home/lpl2025/lpl/inrsteg-final_v1/output1/dual_noise/ismark_v6_size256_msg64/lightning_logs/version_3/checkpoints/ckpt-epoch=04-val_loss=0.1313.ckpt",
    )
    parser.add_argument("--model-module", default="model.ismark_v6_30bit")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-msg", default=None, help="原始消息, 用于对比准确率")
    parser.add_argument("--keep-proxy-env", action="store_true")
    parser.add_argument("--decode-size", type=int, default=None, help="解码输入尺寸, 默认取cfg.img_size")
    parser.add_argument("--crop-sizes", default="128,256,512,1024")
    parser.add_argument("--num-crops-per-size", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260512)
    parser.add_argument("--result-dir", default="./decode_photo_results")
    parser.add_argument("--save-crops", action="store_true", help="保存随机裁剪并用于复查")
    parser.add_argument("--extreme-top-k", type=int, default=50, help="导出 bit_acc 最低和最高的前 K 个裁剪")
    parser.add_argument("--use-qr-correct", action="store_true", help="启用二维码矫正后再裁剪 (YOLO 检测 4 个 QR 标记点)")
    parser.add_argument("--debug-correct", type=int, default=0, metavar="N", help="保存前 N 张矫正调试图到 result_dir")
    parser.add_argument("--gpu", type=int, default=None, help="选择 GPU 编号 (如 --gpu 0 等价于 --device cuda:0)")
    parser.add_argument("--qr-model", default="VerifyWmExactor/Qrlocted/20250729_best.pt", help="YOLO QR 检测模型路径")
    parser.add_argument("--qr-conf", type=float, default=0.6, help="YOLO 检测置信度阈值")
    parser.add_argument("--screen-width", type=int, default=1920, help="显示屏幕宽 (像素)")
    parser.add_argument("--screen-height", type=int, default=1080, help="显示屏幕高 (像素)")
    parser.add_argument("--qr-size", type=int, default=120, help="QR 标记尺寸 (像素)")
    parser.add_argument("--region-size", type=int, default=0, help="矫正目标区域边长, 0=全屏 1920x1080")
    parser.add_argument("--pre-resize-w", type=int, default=0, help="裁剪前缩放宽度, 0=不缩放 (不矫正时建议 full:2000x1125 part:1600x900)")
    parser.add_argument("--pre-resize-h", type=int, default=0, help="裁剪前缩放高度, 0=不缩放")
    parser.add_argument("--edge-margin", type=int, default=0, help="随机裁剪避开边缘像素数, 避免裁到 QR 码 (不矫正时建议 100)")
    parser.add_argument("--no-crop", action="store_true", help="不做裁剪和矫正，直接对整张原图解码")
    cli = parser.parse_args()

    cli.qr_yolo_model = None

    if not cli.keep_proxy_env:
        for k in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
            os.environ.pop(k, None)

    if cli.gpu is not None:
        cli.device = f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu"

    if cli.use_qr_correct:
        if not HAS_CV2 or not HAS_YOLO:
            raise ImportError("--use-qr-correct requires opencv-python-headless and ultralytics. Install: pip install opencv-python-headless ultralytics")
        cli.qr_yolo_model = YOLO(cli.qr_model)
        print(f"[qr] YOLO model loaded from {cli.qr_model}")

    random.seed(cli.random_seed)
    np.random.seed(cli.random_seed)
    torch.manual_seed(cli.random_seed)

    # 如果未显式指定 result-dir，默认保存到 photo-root 下
    if cli.result_dir == "./decode_photo_results":
        suffix = "_no_crop" if cli.no_crop else ""
        if cli.use_qr_correct:
            suffix = "_qr_correct" + (f"_region{cli.region_size}" if cli.region_size > 0 else "")
        cli.result_dir = str(Path(cli.photo_root) / f"decode_photo_results{suffix}")

    ckpt_path = resolve_ckpt(cli.ckpt)
    cfg = load_yaml_args(cli.cfg)
    module = importlib.import_module(cli.model_module)
    model = module.INRMark.load_from_checkpoint(ckpt_path, args=cfg)
    model = model.to(cli.device).eval()

    msg_len = int(cfg.msg_len)
    decode_size = int(cli.decode_size or cfg.img_size)

    if cli.photo:
        decode_single_photo(cli, model, msg_len, decode_size)
    else:
        decode_batch(cli, model, msg_len, decode_size)


if __name__ == "__main__":
    main()
