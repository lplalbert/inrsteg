import argparse
import csv
import importlib
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm


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


def read_img(img_path: str, mode: str = "RGB") -> Image.Image:
    return Image.open(img_path).convert(mode)


def image_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    resample = getattr(Image, "Resampling", Image).BICUBIC
    img = img.resize((size, size), resample=resample)
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor.clamp(0, 1)


def save_tensor_image(tensor: torch.Tensor, path: Path, normalize: bool = False, nrow: int = 8) -> None:
    tensor = tensor.detach().cpu().float()
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if normalize:
        min_val = tensor.min()
        max_val = tensor.max()
        tensor = (tensor - min_val) / (max_val - min_val).clamp_min(1e-12)
    tensor = tensor.clamp(0, 1)

    batch, channels, height, width = tensor.shape
    nrow = max(1, min(nrow, batch))
    ncol = int(np.ceil(batch / nrow))
    grid = torch.zeros(channels, ncol * height, nrow * width)
    for idx in range(batch):
        row = idx // nrow
        col = idx % nrow
        grid[:, row * height : (row + 1) * height, col * width : (col + 1) * width] = tensor[idx]

    if channels == 1:
        arr = (grid.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
    else:
        arr = (grid[:3].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    img.save(path)


def gen_random_msg(n: int) -> torch.Tensor:
    return torch.from_numpy(np.random.choice([0, 1], (n,))).float()


def msg_acc(predict_msg: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
    decoded = torch.round(predict_msg).clamp(0, 1).detach()
    bit_err = torch.abs(decoded - msg).sum() / msg.numel()
    return 1 - bit_err


def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((x - y) ** 2)
    if mse.item() == 0:
        return torch.tensor(100.0, device=x.device)
    return 20 * torch.log10(torch.tensor(1.0, device=x.device) / torch.sqrt(mse))


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, dtype=x.dtype, device=x.device) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    window_2d = (gauss[:, None] @ gauss[None, :]).expand(channels, 1, window_size, window_size)

    padding = window_size // 2
    mu_x = F.conv2d(x, window_2d, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window_2d, padding=padding, groups=channels)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window_2d, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, window_2d, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, window_2d, padding=padding, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return score.mean()


def clip_psnr(container: torch.Tensor, host: torch.Tensor, target_psnr: float, over_clip: bool = False):
    data_range = host.max() - host.min()
    target_mse = data_range**2 / 10 ** (target_psnr / 10)
    residual = container - host
    factor = torch.sqrt(target_mse / torch.mean(residual**2).clamp_min(1e-12))
    if factor.item() >= 1 and not over_clip:
        factor = torch.tensor(1.0, device=container.device)
    return host + residual * factor


def generate_grid_coordinates(top_left, side_length, grid_size=128):
    x = torch.linspace(top_left[0], top_left[0] + side_length, grid_size)
    y = torch.linspace(top_left[1], top_left[1] + side_length, grid_size)
    xv, yv = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xv, yv], dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local INRMark embedding/extraction test with random crop visual metrics."
    )
    parser.add_argument(
        "--cfg",
        default="./config/v6_30bit_true.yaml",
        help="Path to yaml config.",
    )
    parser.add_argument(
        "--ckpt",
        default="./output/ismark_v6_30bit_true/lightning_logs/version_5/checkpoints/last.ckpt",
        help="Checkpoint path. Defaults to cfg.ckpt_path, with local ./output fallback.",
    )
    parser.add_argument(
        "--model-module",
        default="model.ismark_v6_100bit",
        help="Python module that exports INRMark.",
    )
    parser.add_argument(
        "--img-dir",
        default="./output/div2k_imgs",
        help="Directory of clean images.",
    )
    parser.add_argument(
        "--out-dir",
        default="./local_watermark_crop_eval",
        help="Directory for samples, summary.csv and summary.json.",
    )
    parser.add_argument(
        "--stats-dir",
        default=None,
        help="Only summarize an existing result directory that contains summary.csv/crops.csv.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-images", type=int, default=10, help="0 means all images.")
    parser.add_argument("--num-crops", type=int, default=10, help="Random crops per image.")
    parser.add_argument("--image-size", type=int, default=512, help="Square render/eval size.")
    parser.add_argument("--crop-size", type=int, default=128, help="Random crop size.")
    parser.add_argument("--decode-size", type=int, default=None, help="Decoder input size.")
    parser.add_argument("--patch-size", type=int, default=128, help="Render patch size.")
    parser.add_argument("--overlap", type=int, default=0, help="Render patch overlap.")
    parser.add_argument("--fixed-psnr", type=float, default=None, help="Optional PSNR clipping target.")
    parser.add_argument("--message", default=None, help="Binary message. Defaults to random.")
    parser.add_argument(
        "--per-image-msg",
        action="store_true",
        help="Generate a new random message for every image.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-proxy-env",
        action="store_true",
        help="Keep HTTP(S)/ALL proxy environment variables before importing the model.",
    )
    parser.add_argument(
        "--save-limit",
        type=int,
        default=5,
        help="Number of images for which visual samples are saved. 0 disables samples.",
    )
    return parser.parse_args()


def clear_proxy_env() -> Dict[str, str]:
    """Avoid optional Lightning/Gradio imports crashing on unsupported proxy schemes."""
    proxy_keys = [
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ]
    removed = {}
    for key in proxy_keys:
        if key in os.environ:
            removed[key] = os.environ.pop(key)
    return removed


def resolve_ckpt_path(raw_path: str) -> str:
    if raw_path and os.path.exists(raw_path):
        return raw_path

    if raw_path and "/output/" in raw_path:
        local_path = "./output/" + raw_path.split("/output/", 1)[1]
        if os.path.exists(local_path):
            return local_path

    raise FileNotFoundError(
        f"Checkpoint not found: {raw_path}. Pass --ckpt explicitly if it is elsewhere."
    )


def load_model(cfg_path: str, ckpt_path: str, model_module: str, device: str):
    args = load_yaml_args(cfg_path)
    ckpt_path = resolve_ckpt_path(ckpt_path or args.ckpt_path)
    try:
        module = importlib.import_module(model_module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Loading {model_module} requires missing dependency '{exc.name}'. "
            "Run this script in the same environment used for INRMark training/inference."
        ) from exc
    model = module.INRMark.load_from_checkpoint(ckpt_path, args=args)
    model = model.to(device).eval()
    return model, args, ckpt_path


def list_images(img_dir: str, num_images: int) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [
        str(Path(img_dir) / name)
        for name in sorted(os.listdir(img_dir))
        if Path(name).suffix.lower() in exts
    ]
    if num_images and num_images > 0:
        paths = paths[:num_images]
    if not paths:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return paths


def message_to_tensor(msg: str, msg_len: int) -> torch.Tensor:
    if len(msg) != msg_len:
        raise ValueError(f"--message must have exactly {msg_len} bits, got {len(msg)}.")
    if any(ch not in "01" for ch in msg):
        raise ValueError("--message must be a binary string.")
    return torch.tensor([int(ch) for ch in msg], dtype=torch.float32)


def tensor_to_bits(tensor: torch.Tensor, msg_len: int) -> str:
    bits = torch.round(tensor).clamp(0, 1).squeeze(0).detach().cpu()
    return "".join(str(int(x)) for x in bits[:msg_len])


def make_patch_starts(length: int, patch_size: int, overlap: int) -> List[int]:
    if patch_size <= 0:
        raise ValueError("--patch-size must be positive.")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("--overlap must satisfy 0 <= overlap < patch_size.")
    if length <= patch_size:
        return [0]

    stride = patch_size - overlap
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.no_grad()
def render_watermarked_image(
    model,
    img: torch.Tensor,
    coords: torch.Tensor,
    msg: torch.Tensor,
    patch_size: int,
    overlap: int,
    fixed_psnr: float = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, _, h, w = img.shape
    h_starts = make_patch_starts(h, patch_size, overlap)
    w_starts = make_patch_starts(w, patch_size, overlap)
    output = torch.zeros_like(img)
    residual_output = torch.zeros_like(img)
    count_map = torch.zeros_like(img)

    for h_start in h_starts:
        for w_start in w_starts:
            h_end = min(h_start + patch_size, h)
            w_end = min(w_start + patch_size, w)
            patch = img[:, :, h_start:h_end, w_start:w_end]
            patch_coords = coords[:, h_start:h_end, w_start:w_end, :]
            wm_patch, residual = model.render_img(patch_coords, msg, patch)
            if fixed_psnr is not None:
                wm_patch = clip_psnr(wm_patch, patch, fixed_psnr, over_clip=True)
                residual = wm_patch - patch

            output[:, :, h_start:h_end, w_start:w_end] += wm_patch
            residual_output[:, :, h_start:h_end, w_start:w_end] += residual
            count_map[:, :, h_start:h_end, w_start:w_end] += 1

    output = output / count_map.clamp_min(1)
    residual_output = residual_output / count_map.clamp_min(1)
    return output.clamp(0, 1), residual_output


def random_crop_pair(
    clean: torch.Tensor,
    watermarked: torch.Tensor,
    crop_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    _, _, h, w = watermarked.shape
    if h < crop_size or w < crop_size:
        clean_crop = F.interpolate(clean, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        wm_crop = F.interpolate(watermarked, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        return clean_crop, wm_crop, {"top": 0, "left": 0, "size": crop_size, "resized": 1}

    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    clean_crop = clean[:, :, top : top + crop_size, left : left + crop_size]
    wm_crop = watermarked[:, :, top : top + crop_size, left : left + crop_size]
    return clean_crop, wm_crop, {"top": top, "left": left, "size": crop_size, "resized": 0}


@torch.no_grad()
def decode_bits(model, img: torch.Tensor, decode_size: int, msg_len: int) -> Tuple[str, torch.Tensor]:
    if img.shape[-2:] != (decode_size, decode_size):
        img = F.interpolate(img, size=(decode_size, decode_size), mode="bilinear", align_corners=False)
    pred = model.decode_msg(img)
    return tensor_to_bits(pred, msg_len), pred


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
        "std": float(var**0.5),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def print_metric_stats(title: str, stats: Dict[str, Dict[str, float]]) -> None:
    print(f"\n[{title}]")
    print(f"{'metric':<22} {'count':>8} {'mean':>12} {'var':>12} {'std':>12} {'min':>12} {'max':>12}")
    for name, item in stats.items():
        print(
            f"{name:<22} "
            f"{int(item['count']):>8} "
            f"{item['mean']:>12.6f} "
            f"{item['var']:>12.6f} "
            f"{item['std']:>12.6f} "
            f"{item['min']:>12.6f} "
            f"{item['max']:>12.6f}"
        )


def read_csv_numeric_columns(path: Path, columns: List[str]) -> Dict[str, List[float]]:
    values = {column: [] for column in columns}
    if not path.exists():
        return values

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column in columns:
                raw_value = row.get(column)
                if raw_value in (None, ""):
                    continue
                values[column].append(float(raw_value))
    return values


def summarize_result_dir(result_dir: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    result_path = Path(result_dir)
    summary_values = read_csv_numeric_columns(
        result_path / "summary.csv",
        [
            "full_psnr",
            "full_ssim",
            "full_bit_acc",
            "crop_psnr_mean",
            "crop_ssim_mean",
            "crop_bit_acc_mean",
            "crop_bit_acc_min",
        ],
    )
    crop_values = read_csv_numeric_columns(result_path / "crops.csv", ["psnr", "ssim", "bit_acc"])

    summary_stats = {key: metric_stats(value) for key, value in summary_values.items()}
    crop_stats = {
        "crop_psnr": metric_stats(crop_values["psnr"]),
        "crop_ssim": metric_stats(crop_values["ssim"]),
        "crop_bit_acc": metric_stats(crop_values["bit_acc"]),
        "crop_ber": metric_stats([1 - value for value in crop_values["bit_acc"]]),
    }

    print_metric_stats("summary.csv per-image stats", summary_stats)
    print_metric_stats("crops.csv per-crop stats", crop_stats)
    return {"summary": summary_stats, "crops": crop_stats}


def save_visual_samples(
    sample_dir: Path,
    stem: str,
    clean: torch.Tensor,
    watermarked: torch.Tensor,
    residual: torch.Tensor,
    first_clean_crop: torch.Tensor,
    first_wm_crop: torch.Tensor,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_tensor_image(clean, sample_dir / f"{stem}_clean.png")
    save_tensor_image(watermarked, sample_dir / f"{stem}_watermarked.png")
    save_tensor_image((watermarked - clean).abs() * 10, sample_dir / f"{stem}_diff_x10.png")
    save_tensor_image(residual, sample_dir / f"{stem}_residual.png", normalize=True)
    save_tensor_image(
        torch.cat([first_clean_crop, first_wm_crop, (first_wm_crop - first_clean_crop).abs() * 10], dim=0),
        sample_dir / f"{stem}_crop_clean_wm_diffx10.png",
        nrow=3,
    )


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cli = parse_args()
    if cli.stats_dir:
        summarize_result_dir(cli.stats_dir)
        return
    if cli.num_crops <= 0:
        raise ValueError("--num-crops must be positive for random-crop evaluation.")
    if not cli.keep_proxy_env:
        clear_proxy_env()
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    model, cfg, ckpt_path = load_model(cli.cfg, cli.ckpt, cli.model_module, cli.device)
    out_dir = Path(cli.out_dir) / time.strftime("%Y%m%d-%H%M%S")
    sample_dir = out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    msg_len = int(cfg.msg_len)
    decode_size = int(cli.decode_size or cfg.img_size)
    fixed_msg = None
    if cli.message:
        fixed_msg = message_to_tensor(cli.message, msg_len)
    elif not cli.per_image_msg:
        fixed_msg = gen_random_msg(msg_len)

    coords = generate_grid_coordinates((-1, -1), 2, cli.image_size).unsqueeze(0).to(cli.device)
    image_paths = list_images(cli.img_dir, cli.num_images)

    rows: List[Dict] = []
    crop_rows: List[Dict] = []
    summary_values = {
        "full_psnr": [],
        "full_ssim": [],
        "full_acc": [],
        "crop_psnr": [],
        "crop_ssim": [],
        "crop_acc": [],
    }

    for img_index, img_path in enumerate(tqdm(image_paths, desc="Evaluating images")):
        msg = gen_random_msg(msg_len) if cli.per_image_msg else fixed_msg.clone()
        msg = msg.unsqueeze(0).to(cli.device)
        msg_str = tensor_to_bits(msg, msg_len)

        clean = read_img(img_path)
        clean = image_to_tensor(clean, cli.image_size).unsqueeze(0).to(cli.device)
        watermarked, residual = render_watermarked_image(
            model=model,
            img=clean,
            coords=coords,
            msg=msg,
            patch_size=cli.patch_size,
            overlap=cli.overlap,
            fixed_psnr=cli.fixed_psnr,
        )

        full_psnr = float(psnr(watermarked, clean).item())
        full_ssim = float(ssim(watermarked, clean).item())
        full_bits, full_pred = decode_bits(model, watermarked, decode_size, msg_len)
        full_acc = float(msg_acc(full_pred, msg).item())
        summary_values["full_psnr"].append(full_psnr)
        summary_values["full_ssim"].append(full_ssim)
        summary_values["full_acc"].append(full_acc)

        crop_accs = []
        crop_psnrs = []
        crop_ssims = []
        first_clean_crop = None
        first_wm_crop = None
        first_crop_bits = ""
        first_crop_meta = {}

        for crop_index in range(cli.num_crops):
            clean_crop, wm_crop, crop_meta = random_crop_pair(clean, watermarked, cli.crop_size)
            crop_bits, crop_pred = decode_bits(model, wm_crop, decode_size, msg_len)
            crop_acc = float(msg_acc(crop_pred, msg).item())
            crop_psnr = float(psnr(wm_crop, clean_crop).item())
            crop_ssim = float(ssim(wm_crop, clean_crop).item())

            crop_accs.append(crop_acc)
            crop_psnrs.append(crop_psnr)
            crop_ssims.append(crop_ssim)
            summary_values["crop_acc"].append(crop_acc)
            summary_values["crop_psnr"].append(crop_psnr)
            summary_values["crop_ssim"].append(crop_ssim)

            if crop_index == 0:
                first_clean_crop = clean_crop
                first_wm_crop = wm_crop
                first_crop_bits = crop_bits
                first_crop_meta = crop_meta

            crop_rows.append(
                {
                    "image": os.path.basename(img_path),
                    "crop_index": crop_index,
                    "top": crop_meta["top"],
                    "left": crop_meta["left"],
                    "crop_size": crop_meta["size"],
                    "resized": crop_meta["resized"],
                    "psnr": crop_psnr,
                    "ssim": crop_ssim,
                    "bit_acc": crop_acc,
                    "target_msg": msg_str,
                    "decoded_msg": crop_bits,
                }
            )

        row = {
            "image": os.path.basename(img_path),
            "target_msg": msg_str,
            "full_decoded_msg": full_bits,
            "first_crop_decoded_msg": first_crop_bits,
            "full_psnr": full_psnr,
            "full_ssim": full_ssim,
            "full_bit_acc": full_acc,
            "crop_psnr_mean": mean(crop_psnrs),
            "crop_ssim_mean": mean(crop_ssims),
            "crop_bit_acc_mean": mean(crop_accs),
            "crop_bit_acc_min": min(crop_accs) if crop_accs else 0.0,
            "first_crop_top": first_crop_meta.get("top", 0),
            "first_crop_left": first_crop_meta.get("left", 0),
        }
        rows.append(row)

        if cli.save_limit > 0 and img_index < cli.save_limit:
            stem = f"{img_index:03d}_{Path(img_path).stem[:32]}"
            save_visual_samples(
                sample_dir,
                stem,
                clean.detach().cpu(),
                watermarked.detach().cpu(),
                residual.detach().cpu(),
                first_clean_crop.detach().cpu(),
                first_wm_crop.detach().cpu(),
            )

    write_csv(out_dir / "summary.csv", rows)
    write_csv(out_dir / "crops.csv", crop_rows)

    stats = {key: metric_stats(value) for key, value in summary_values.items()}
    summary = {
        "cfg": cli.cfg,
        "ckpt": ckpt_path,
        "model_module": cli.model_module,
        "img_dir": cli.img_dir,
        "num_images": len(image_paths),
        "num_crops_per_image": cli.num_crops,
        "image_size": cli.image_size,
        "crop_size": cli.crop_size,
        "decode_size": decode_size,
        "msg_len": msg_len,
        "fixed_message": tensor_to_bits(fixed_msg.unsqueeze(0), msg_len) if fixed_msg is not None else None,
        "metrics": {key: mean(value) for key, value in summary_values.items()},
        "metric_stats": stats,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print_metric_stats("evaluation stats", stats)
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
