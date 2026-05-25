"""
拍照测试脚本：批量生成模板 + 全屏显示

用法:
  # 第一步：批量生成 10 个模板 (种子 0-9)
  python display_for_camera.py --gen-batch

  # 第二步：显示某个模板 (默认 seed=0)
  python display_for_camera.py --seed 0

  # 换模板 / 换底图
  python display_for_camera.py --seed 5 --image-index 3

  # 预览模式 (不全屏)
  python display_for_camera.py --seed 0 --no-fullscreen
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


# ═══════════════════════════════════════════════════════════════
# util
# ═══════════════════════════════════════════════════════════════

class DotDict(dict):
    def __getattr__(self, key): return self.get(key, None)
    def __setattr__(self, key, value): self[key] = value


def load_yaml_args(path: str) -> DotDict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    args = DotDict()
    for key, value in data.items():
        args[key] = DotDict(value) if isinstance(value, dict) else value
    return args


def image_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).clamp(0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    t = tensor.detach().cpu().clamp(0, 1)
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def generate_grid_coordinates(top_left, side_length, grid_size):
    x = torch.linspace(top_left[0], top_left[0] + side_length, grid_size)
    y = torch.linspace(top_left[1], top_left[1] + side_length, grid_size)
    xv, yv = torch.meshgrid(x, y, indexing="ij")
    return torch.stack([xv, yv], dim=-1)


def make_patch_starts(length: int, patch_size: int, overlap: int):
    if length <= patch_size:
        return [0]
    stride = patch_size - overlap
    starts = list(range(0, length - patch_size + 1, stride))
    final = length - patch_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def crop_center(tensor, target_h, target_w):
    _, _, h, w = tensor.shape
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return tensor[:, :, top:top + target_h, left:left + target_w]


def load_model(cfg_path, ckpt_path, model_module, device):
    args = load_yaml_args(cfg_path)
    module = importlib.import_module(model_module)
    model = module.INRMark.load_from_checkpoint(ckpt_path, args=args)
    return model.to(device).eval(), args


def resolve_ckpt(raw: str) -> str:
    if raw and os.path.exists(raw):
        return raw
    if raw and "/output/" in raw:
        local = "./output/" + raw.split("/output/", 1)[1]
        if os.path.exists(local):
            return local
    raise FileNotFoundError(f"Checkpoint not found: {raw}")


# ═══════════════════════════════════════════════════════════════
# watermark
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def render_template(model, coords, msg, patch_size, overlap):
    _, h, w, _ = coords.shape
    h_starts = make_patch_starts(h, patch_size, overlap)
    w_starts = make_patch_starts(w, patch_size, overlap)
    output = torch.zeros((1, 3, h, w), device=coords.device)
    count = torch.zeros_like(output)

    for hs in h_starts:
        for ws in w_starts:
            he = min(hs + patch_size, h)
            we = min(ws + patch_size, w)
            wm_patch, _ = model.render_img(coords[:, hs:he, ws:we, :], msg)
            output[:, :, hs:he, ws:we] += wm_patch
            count[:, :, hs:he, ws:we] += 1

    return output / count.clamp_min(1)


# ═══════════════════════════════════════════════════════════════
# mode: 批量生成
# ═══════════════════════════════════════════════════════════════

def mode_gen_batch(cli):
    """种子 0~9, 每个种子确定性地生成不同的随机消息 → 模板"""
    ckpt_path = resolve_ckpt(cli.ckpt)
    model, cfg = load_model(cli.cfg, ckpt_path, cli.model_module, cli.device)
    msg_len = int(cfg.msg_len)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 坐标网格复用, 只算一次
    coords = generate_grid_coordinates((-1, -1), 2, cli.template_size).unsqueeze(0).to(cli.device)

    manifest = []
    seeds = list(range(cli.batch_start, cli.batch_start + cli.batch_count))

    for seed in seeds:
        # 伪随机可复现: 种子 → 固定bit序列
        torch.manual_seed(seed)
        np.random.seed(seed)
        msg_tensor = torch.from_numpy(np.random.choice([0, 1], (msg_len,))).float()
        msg_str = "".join(str(int(x)) for x in msg_tensor)
        msg = msg_tensor.unsqueeze(0).to(cli.device)

        print(f"seed={seed}  msg={msg_str}  rendering ...")

        template = render_template(model, coords, msg, cli.patch_size, cli.overlap)

        # 保存
        torch.save(template.cpu(), out_dir / f"template_seed{seed}.pt")
        (out_dir / f"message_seed{seed}.txt").write_text(msg_str + "\n")

        # 归一化可视化 (跳过 template_vis, 太暗看不清)
        t = template.squeeze(0)
        t_norm = (t - t.min()) / (t.max() - t.min()).clamp_min(1e-12)
        tensor_to_pil(t_norm.unsqueeze(0)).save(out_dir / f"template_seed{seed}_norm.png")

        # 裁剪到显示尺寸，用于 overlay
        dh, dw = cli.display_h, cli.display_w
        t_crop = crop_center(template, dh, dw).squeeze(0)
        t_crop_norm = (t_crop - t_crop.min()) / (t_crop.max() - t_crop.min()).clamp_min(1e-12)
        tensor_to_pil(t_crop_norm.unsqueeze(0)).save(out_dir / f"template_seed{seed}_crop.png")

        # 裁剪到 region 尺寸，用于局部区域模式
        if cli.region_size > 0:
            rs = cli.region_size
            t_region = crop_center(template, rs, rs).squeeze(0)
            t_region_norm = (t_region - t_region.min()) / (t_region.max() - t_region.min()).clamp_min(1e-12)
            tensor_to_pil(t_region_norm.unsqueeze(0)).save(out_dir / f"template_seed{seed}_region{rs}.png")

        manifest.append({"seed": seed, "message": msg_str})

    # 清单
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n批量生成完毕: {len(manifest)} 个模板 → {out_dir}/")
    print(f"  清单: manifest.json")
    print(f"  下次显示: python display_for_camera.py --seed 0")


# ═══════════════════════════════════════════════════════════════
# mode: 显示
# ═══════════════════════════════════════════════════════════════

def mode_display(cli):
    import subprocess
    import time

    out_dir = Path(cli.out_dir)
    script_dir = Path(__file__).parent
    show_content_dir = Path(r"C:\Users\24976\Desktop\code\show_content")

    # ── 水印 PNG 路径（始终用全屏尺寸，overlay 覆盖整个屏幕）──
    if cli.load_template:
        wm_image_path = os.path.abspath(cli.load_template)
    else:
        wm_image_path = str(out_dir / f"template_seed{cli.seed}_crop.png")
    if not os.path.exists(wm_image_path):
        raise FileNotFoundError(
            f"水印图不存在: {wm_image_path}\n"
            f"  请先生成: python display_for_camera.py --gen-batch")

    # 消息
    if cli.load_template:
        msg_path = None
    else:
        msg_path = out_dir / f"message_seed{cli.seed}.txt"
        if not msg_path.exists():
            msg_path = None

    py = sys.executable
    qr_size = 0 if cli.no_qr else cli.qr_size

    print(f"seed: {cli.seed}")
    if msg_path:
        print(f"message: {msg_path.read_text().strip()}")
    print(f"region_size: {cli.region_size}  qr_size: {qr_size}  alpha: {cli.alpha_scale}")

    # ── 启动播放器 (player.py) ──
    player_cmd = [
        py, str(script_dir / "player.py"),
        "--image-folder", cli.img_dir,
        "--screen-index", str(cli.screen_index),
        "--turn-time", str(cli.turn_time),
        "--qr-size", str(qr_size),
    ]
    if cli.region_size > 0:
        player_cmd += ["--region-size", str(cli.region_size)]
    if cli.no_qr:
        player_cmd.append("--no-qr")
    print("Starting player:", " ".join(player_cmd))
    player_proc = subprocess.Popen(player_cmd)

    # ── 等待后启动 overlay（水印始终全屏）──
    time.sleep(max(0.0, cli.delay))
    overlay_template_dir = show_content_dir / "screen_templates" / "v0"
    overlay_template_dir.mkdir(parents=True, exist_ok=True)
    overlay_template_path = overlay_template_dir / f"camera_seed{cli.seed}.png"
    import shutil
    shutil.copy2(wm_image_path, overlay_template_path)

    overlay_cmd = [
        py, str(show_content_dir / "test6_stable_overlay.py"),
        "--alpha", str(cli.alpha_scale),
        "--template", str(overlay_template_path),
        "--guard-interval-ms", "500",
        "--boot-guard-ticks", "20",
    ]
    print("Starting overlay:", " ".join(overlay_cmd))
    overlay_proc = subprocess.Popen(overlay_cmd)

    print("Running. Press Ctrl+C to stop.")

    try:
        while True:
            if player_proc.poll() is not None:
                print("Player exited.")
                break
            if overlay_proc is not None and overlay_proc.poll() is not None:
                print("Overlay exited.")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        for proc in [player_proc, overlay_proc]:
            if proc is not None and proc.poll() is None:
                proc.terminate()

    # 解码提示
    ckpt_path = resolve_ckpt(cli.ckpt)
    region_flag = f" --region-size {cli.region_size}" if cli.region_size > 0 else ""
    print("\n" + "=" * 60)
    print("拍照后解码:")
    print(f"  python decode_photo.py --photo-root test_photo/v0_30_128{region_flag}")
    print(f"  --cfg {cli.cfg} --ckpt {ckpt_path} --model-module {cli.model_module}")
    if msg_path:
        print(f"  --target-msg {msg_path.read_text().strip()}")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="水印模板批量生成 & 全屏拍照测试")
    # 模型
    parser.add_argument("--cfg", default="./config/v6_size256_msg64.yaml")
    parser.add_argument("--ckpt",
        default="/home/lpl2025/lpl/inrsteg-final_v1/output1/dual_noise/ismark_v6_size256_msg64/lightning_logs/version_3/checkpoints/ckpt-epoch=04-val_loss=0.1313.ckpt")
    parser.add_argument("--model-module", default="model.ismark_v6_30bit")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    # 图像
    parser.add_argument("--img-dir", default=r"C:\Users\24976\Desktop\code\show_content\raw_input")
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--out-dir", default="./display_test3_256_64_dual")
    # 模板
    parser.add_argument("--template-size", type=int, default=2048)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--display-w", type=int, default=1920)
    parser.add_argument("--display-h", type=int, default=1080)
    # 批量生成
    parser.add_argument("--gen-batch", action="store_true")
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--batch-count", type=int, default=10)
    # 显示
    parser.add_argument("--seed", type=int, default=0, help="加载哪个种子的模板")
    parser.add_argument("--load-template", default=None, help="或直接指定模板路径")
    parser.add_argument("--alpha-scale", type=float, default=0.02)
    parser.add_argument("--turn-time", type=int, default=1000)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--screen-index", type=int, default=0)
    parser.add_argument("--qr-size", type=int, default=120)
    parser.add_argument("--no-qr", action="store_true", help="不显示二维码")
    parser.add_argument("--region-size", type=int, default=0, help="居中区域边长，0=全屏")
    parser.add_argument("--keep-proxy-env", action="store_true")
    cli = parser.parse_args()

    if not cli.keep_proxy_env:
        for k in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
            os.environ.pop(k, None)

    if cli.gen_batch:
        mode_gen_batch(cli)
    else:
        mode_display(cli)


if __name__ == "__main__":
    main()
