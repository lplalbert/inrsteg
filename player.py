"""
图片播放器：支持 --no-qr、屏幕四角二维码、--region-size 居中区域模式
"""

import argparse
import glob
import os
import random
import sys
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk
from screeninfo import get_monitors

QR_IMAGE_PATH = r"qrmark\qr_loc_mark.png"
SHOW_CONTENT_QR_PATH = r"C:\Users\24976\Desktop\code\show_content\qrmark\qr_loc_mark.png"


def resolve_qr_path():
    if os.path.exists(QR_IMAGE_PATH):
        return QR_IMAGE_PATH
    if os.path.exists(SHOW_CONTENT_QR_PATH):
        return SHOW_CONTENT_QR_PATH
    raise FileNotFoundError(f"QR marker not found: {QR_IMAGE_PATH} or {SHOW_CONTENT_QR_PATH}")


def get_image_files_from_folder(folder_path):
    image_patterns = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(folder_path, pattern)))

    def sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem.lower())

    image_files.sort(key=sort_key)
    return image_files


def add_qr_at_screen_corners(image_rgb, qr_size=120, margin=0):
    """二维码放在屏幕四个角"""
    qr_path = resolve_qr_path()
    qr = cv2.imread(qr_path)
    if qr is None:
        raise ValueError(f"Cannot read QR marker: {qr_path}")
    qr = cv2.cvtColor(qr, cv2.COLOR_BGR2RGB)
    qr = cv2.resize(qr, (qr_size, qr_size))

    h, w = image_rgb.shape[:2]
    positions = [
        (margin, margin),
        (w - qr_size - margin, margin),
        (margin, h - qr_size - margin),
        (w - qr_size - margin, h - qr_size - margin),
    ]

    centers = []
    for px, py in positions:
        image_rgb[py:py + qr_size, px:px + qr_size] = qr
        centers.append((px + qr_size // 2, py + qr_size // 2))

    return image_rgb, centers


def add_qr_at_region_corners(image_rgb, region_left, region_top, region_size, qr_size=120):
    """二维码放在区域四角外侧，QR中心对准区域角点"""
    qr_path = resolve_qr_path()
    qr = cv2.imread(qr_path)
    if qr is None:
        raise ValueError(f"Cannot read QR marker: {qr_path}")
    qr = cv2.cvtColor(qr, cv2.COLOR_BGR2RGB)
    qr = cv2.resize(qr, (qr_size, qr_size))

    h, w = image_rgb.shape[:2]
    half_qr = qr_size // 2

    # QR 中心 = 区域四角，QR 块向区域外侧偏移
    corners = [
        (region_left, region_top),                               # TL
        (region_left + region_size, region_top),                 # TR
        (region_left + region_size, region_top + region_size),   # BR
        (region_left, region_top + region_size),                 # BL
    ]
    centers = []
    for cx, cy in corners:
        px = max(0, min(cx - half_qr, w - qr_size))
        py = max(0, min(cy - half_qr, h - qr_size))
        image_rgb[py:py + qr_size, px:px + qr_size] = qr
        centers.append((cx, cy))
    return image_rgb, centers


def build_frame(image_path, screen_w, screen_h, seq_id, qr_size=0, region_size=0):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (screen_w, screen_h))
    qr_centers = []

    if region_size > 0 and qr_size > 0:
        # 全屏显示图片，仅把 QR 放在 region 四角外侧
        region_left = (screen_w - region_size) // 2
        region_top = (screen_h - region_size) // 2
        img_rgb, qr_centers = add_qr_at_region_corners(img_rgb, region_left, region_top, region_size, qr_size)
    elif qr_size > 0:
        img_rgb, qr_centers = add_qr_at_screen_corners(img_rgb, qr_size=qr_size, margin=0)

    text = str(seq_id)
    text_x = 24
    text_y = qr_size + 56 if qr_size > 0 else 56
    cv2.putText(img_rgb, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 6, cv2.LINE_AA)
    cv2.putText(img_rgb, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 2, cv2.LINE_AA)

    return Image.fromarray(img_rgb), qr_centers


def move_window_to_screen(window, screen_index):
    monitors = get_monitors()
    if screen_index >= len(monitors):
        raise ValueError(f"screen_index {screen_index} out of range, total monitors={len(monitors)}")
    monitor = monitors[screen_index]
    window.geometry(f"{monitor.width}x{monitor.height}+{monitor.x}+{monitor.y}")
    return monitor.width, monitor.height


def configure_display_window(window):
    window.overrideredirect(True)
    window.lift()
    window.focus_force()

    def on_escape(event=None):
        window.destroy()
        sys.exit(0)

    window.bind("<Escape>", on_escape)
    window.bind_all("<Escape>", on_escape)
    window.protocol("WM_DELETE_WINDOW", on_escape)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="图片播放器")
    parser.add_argument("--image-folder", default=None)
    parser.add_argument("--screen-index", type=int, default=0)
    parser.add_argument("--turn-time", type=int, default=3000)
    parser.add_argument("--qr-size", type=int, default=120)
    parser.add_argument("--no-qr", action="store_true", help="不显示二维码")
    parser.add_argument("--region-size", type=int, default=0, help="居中区域边长，QR放在区域四角，0=屏幕四角")
    parser.add_argument("--shuffle", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    image_folder = args.image_folder or r"C:\Users\24976\Desktop\code\show_content\raw_input"
    image_files = get_image_files_from_folder(image_folder)

    if not image_files:
        raise ValueError(f"No images found in: {image_folder}")

    if args.shuffle:
        random.shuffle(image_files)

    qr_size = 0 if args.no_qr else args.qr_size

    root = tk.Tk()
    screen_w, screen_h = move_window_to_screen(root, args.screen_index)
    configure_display_window(root)

    label = tk.Label(root)
    label.pack(fill=tk.BOTH, expand=True)

    image_index = 0

    def change_image():
        global image_index
        image_path = image_files[image_index]
        seq_id = image_index
        image_index = (image_index + 1) % len(image_files)

        frame, _ = build_frame(
            image_path=image_path,
            screen_w=screen_w,
            screen_h=screen_h,
            seq_id=seq_id,
            qr_size=qr_size,
            region_size=args.region_size,
        )

        photo = ImageTk.PhotoImage(frame)
        label.config(image=photo)
        label.image = photo
        frame.close()

        root.after(args.turn_time, change_image)

    region_info = f" qr_at_region={args.region_size}x{args.region_size}" if args.region_size > 0 else ""
    print(
        f"Playback: {len(image_files)} images, screen={screen_w}x{screen_h}, "
        f"qr={'region-corners' if args.region_size > 0 else 'screen-corners' if qr_size > 0 else 'off'}{region_info}"
    )

    change_image()
    root.mainloop()
