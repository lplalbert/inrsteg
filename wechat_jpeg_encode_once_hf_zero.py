"""
在 `wechat_jpeg_encode_once.py` 基础上增加：**Pillow 单次 JPEG 编码前**，对图像做「高频强制清零」预处理。

预处理（与 JPEG 分量一致）：
  RGB → YCbCr(BT.601) → 4:2:0（Cb₂/Cr₂）→ 对 **Y** 与 **Cb₂、Cr₂** 各平面做 8×8 分块
  → 减 128 → JPEG 型 DCT → 按 **JPEG 之字形**保留前 `--zigzag-keep` 个系数，其余系数置 0
  → IDCT → 加 128 → 裁回尺寸 → 色度上采样 → YCbCr→RGB

随后仍只调用 **一次** `Image.save(..., JPEG, subsampling=2, qtables=...)`，避免「先高质量 JPEG 再比」的二次码流问题。

`--zigzag-keep=64` 等价于不做高频清零（仅单次 Pillow 编码）。

依赖：numpy、Pillow。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

JPEG_ZZ_ROWMAJOR: Tuple[int, ...] = (
    0,
    1,
    8,
    16,
    9,
    2,
    3,
    10,
    17,
    24,
    32,
    25,
    18,
    11,
    4,
    5,
    12,
    19,
    26,
    33,
    40,
    48,
    41,
    34,
    27,
    20,
    13,
    6,
    7,
    14,
    21,
    28,
    35,
    42,
    49,
    56,
    57,
    50,
    43,
    36,
    29,
    22,
    15,
    23,
    30,
    37,
    44,
    51,
    58,
    59,
    52,
    45,
    38,
    31,
    39,
    46,
    53,
    60,
    61,
    54,
    47,
    55,
    62,
    63,
)

Q_LUMINANCE_DEFAULT = np.array(
    [
        [13, 9, 8, 13, 19, 32, 41, 49],
        [10, 10, 11, 15, 21, 46, 48, 44],
        [11, 10, 13, 19, 32, 46, 55, 45],
        [11, 14, 18, 23, 41, 70, 64, 50],
        [14, 18, 30, 45, 54, 87, 82, 62],
        [19, 28, 44, 51, 65, 83, 90, 74],
        [39, 51, 62, 70, 82, 97, 96, 81],
        [58, 74, 76, 78, 90, 80, 82, 79],
    ],
    dtype=np.int32,
)

Q_CHROMINANCE_DEFAULT = np.array(
    [
        [14, 14, 19, 38, 79, 79, 79, 79],
        [14, 17, 21, 53, 79, 79, 79, 79],
        [19, 21, 45, 79, 79, 79, 79, 79],
        [38, 53, 79, 79, 79, 79, 79, 79],
        [79, 79, 79, 79, 79, 79, 79, 79],
        [79, 79, 79, 79, 79, 79, 79, 79],
        [79, 79, 79, 79, 79, 79, 79, 79],
        [79, 79, 79, 79, 79, 79, 79, 79],
    ],
    dtype=np.int32,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def q_rowmajor_to_zigzag_list(q8: np.ndarray) -> List[int]:
    flat = np.asarray(q8, dtype=np.int32).ravel(order="C")
    return [int(flat[i]) for i in JPEG_ZZ_ROWMAJOR]


def load_qtables_zigzag_from_jpeg(path: Path) -> Tuple[List[int], List[int]]:
    im = Image.open(path)
    q = getattr(im, "quantization", None)
    im.close()
    if not q or 0 not in q:
        raise ValueError("无法读取 DQT（非 JPEG 或缺少表 0）")
    lum = [int(x) for x in q[0]]
    if len(lum) != 64:
        raise ValueError("亮度 DQT 长度须为 64")
    if 1 in q:
        ch = [int(x) for x in q[1]]
    else:
        ch = list(lum)
    if len(ch) != 64:
        raise ValueError("色度 DQT 长度须为 64")
    return lum, ch


def default_qtables_zigzag() -> Tuple[List[int], List[int]]:
    return (
        q_rowmajor_to_zigzag_list(Q_LUMINANCE_DEFAULT),
        q_rowmajor_to_zigzag_list(Q_CHROMINANCE_DEFAULT),
    )


def rgb_to_ycbcr_bt601(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return y, cb, cr


def ycbcr_to_rgb_bt601(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def chroma_420_downsample(cb: np.ndarray, cr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = cb.shape
    ch, cw = (h + 1) // 2, (w + 1) // 2
    cb2 = np.zeros((ch, cw), dtype=np.float32)
    cr2 = np.zeros((ch, cw), dtype=np.float32)
    for i in range(ch):
        for j in range(cw):
            i0, i1 = i * 2, min(i * 2 + 2, h)
            j0, j1 = j * 2, min(j * 2 + 2, w)
            cb2[i, j] = float(cb[i0:i1, j0:j1].mean())
            cr2[i, j] = float(cr[i0:i1, j0:j1].mean())
    return cb2, cr2


def chroma_420_upsample(cb2: np.ndarray, cr2: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    cb_up = np.repeat(np.repeat(cb2, 2, axis=0), 2, axis=1)[:h, :w]
    cr_up = np.repeat(np.repeat(cr2, 2, axis=0), 2, axis=1)[:h, :w]
    return cb_up.astype(np.float32), cr_up.astype(np.float32)


def pad_to_multiple_of_8(x: np.ndarray) -> np.ndarray:
    h, w = x.shape
    ph = int(math.ceil(h / 8.0) * 8)
    pw = int(math.ceil(w / 8.0) * 8)
    if ph == h and pw == w:
        return x
    out = np.zeros((ph, pw), dtype=x.dtype)
    out[:h, :w] = x
    if pw > w:
        out[:, w:pw] = np.expand_dims(out[:, w - 1], axis=1)
    if ph > h:
        out[h:ph, :] = np.expand_dims(out[h - 1, :], axis=0)
    return out


def dct2_jpeg(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    out = np.zeros((8, 8), dtype=np.float64)
    for u in range(8):
        for v in range(8):
            s = 0.0
            for i in range(8):
                for j in range(8):
                    s += x[i, j] * math.cos((2 * i + 1) * u * math.pi / 16.0) * math.cos((2 * j + 1) * v * math.pi / 16.0)
            cu = 1.0 / math.sqrt(2.0) if u == 0 else 1.0
            cv = 1.0 / math.sqrt(2.0) if v == 0 else 1.0
            out[u, v] = 0.25 * cu * cv * s
    return out


def idct2_jpeg(s: np.ndarray) -> np.ndarray:
    s = s.astype(np.float64)
    out = np.zeros((8, 8), dtype=np.float64)
    for i in range(8):
        for j in range(8):
            acc = 0.0
            for u in range(8):
                for v in range(8):
                    cu = 1.0 / math.sqrt(2.0) if u == 0 else 1.0
                    cv = 1.0 / math.sqrt(2.0) if v == 0 else 1.0
                    acc += (
                        cu
                        * cv
                        * s[u, v]
                        * math.cos((2 * i + 1) * u * math.pi / 16.0)
                        * math.cos((2 * j + 1) * v * math.pi / 16.0)
                    )
            out[i, j] = 0.25 * acc
    return out


def _zero_hf_zigzag(d: np.ndarray, zigzag_keep: int) -> None:
    """就地修改 8×8 DCT 系数：保留之字形前 zigzag_keep 个，其余置 0。"""
    k0 = int(np.clip(zigzag_keep, 1, 64))
    for k in range(k0, 64):
        idx = JPEG_ZZ_ROWMAJOR[k]
        u, v = idx // 8, idx % 8
        d[u, v] = 0.0


def block_dct_hf_zero_idct(plane: np.ndarray, zigzag_keep: int) -> np.ndarray:
    h, w = plane.shape
    padded = pad_to_multiple_of_8(plane.astype(np.float64))
    ph, pw = padded.shape
    recon = np.zeros_like(padded)
    for bi in range(0, ph, 8):
        for bj in range(0, pw, 8):
            blk = padded[bi : bi + 8, bj : bj + 8] - 128.0
            d = dct2_jpeg(blk)
            _zero_hf_zigzag(d, zigzag_keep)
            recon[bi : bi + 8, bj : bj + 8] = idct2_jpeg(d) + 128.0
    return np.clip(recon[:h, :w], 0.0, 255.0).astype(np.float32)


def preprocess_hf_zero_rgb(rgb: np.ndarray, zigzag_keep: int) -> np.ndarray:
    """uint8 RGB (H,W,3) → 高频清零后 uint8 RGB。"""
    y, cb, cr = rgb_to_ycbcr_bt601(rgb)
    cb2, cr2 = chroma_420_downsample(cb, cr)
    hh, ww = y.shape
    yq = block_dct_hf_zero_idct(y, zigzag_keep)
    cbq = block_dct_hf_zero_idct(cb2, zigzag_keep)
    crq = block_dct_hf_zero_idct(cr2, zigzag_keep)
    cb_up, cr_up = chroma_420_upsample(cbq, crq, hh, ww)
    return ycbcr_to_rgb_bt601(yq, cb_up, cr_up)


def save_jpeg_once(
    rgb: Image.Image,
    out_path: Path,
    lum_z: List[int],
    chr_z: List[int],
    quality: int,
) -> Path:
    out_path = out_path.with_suffix(".jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(
        out_path,
        format="JPEG",
        quality=int(quality),
        subsampling=2,
        qtables=[lum_z, chr_z],
        optimize=True,
        progressive=False,
    )
    return out_path


def resize_to_array(path: Path, tw: int, th: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
        return np.asarray(im)


def iter_image_paths(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


def default_out_path(src: Path, out_dir: Optional[Path], suffix: str) -> Path:
    base = src.stem + suffix + ".jpg"
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / base
    return src.with_name(base)


def process_one(
    src: Path,
    out_path: Path,
    tw: int,
    th: int,
    lum_z: List[int],
    chr_z: List[int],
    quality: int,
    zigzag_keep: int,
) -> Path:
    arr = resize_to_array(src, tw, th)
    if zigzag_keep < 64:
        arr = preprocess_hf_zero_rgb(arr, zigzag_keep)
    pil = Image.fromarray(arr, mode="RGB")
    return save_jpeg_once(pil, out_path, lum_z, chr_z, quality)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pillow 单次 JPEG 前：YCbCr+420 分块 DCT 之字形高频清零，再编码"
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", "-i", type=str, help="输入图片")
    g.add_argument("--dir", "-d", type=str, help="输入文件夹")
    ap.add_argument("--out", "-o", type=str, default="", help="单图输出 .jpg")
    ap.add_argument("--out-dir", type=str, default="", help="目录输出（默认 输入目录/wechat_jpg_hf_zero）")
    ap.add_argument("--suffix", type=str, default="_wechat_jpg_hf", help="目录模式文件名后缀")
    ap.add_argument("--tw", type=int, default=1706, help="目标宽")
    ap.add_argument("--th", type=int, default=1279, help="目标高")
    ap.add_argument(
        "--zigzag-keep",
        type=int,
        default=21,
        help="JPEG 之字形下保留的系数个数 1..64（含 DC）；64=不清零，仅 Pillow 编码",
    )
    ap.add_argument("--quality", "-q", type=int, default=60, help="JPEG quality")
    ap.add_argument(
        "--reference-jpeg",
        type=str,
        default="",
        help="从该 JPEG 读取 DQT 覆盖内嵌表",
    )
    args = ap.parse_args()

    zk = int(args.zigzag_keep)
    if zk < 1 or zk > 64:
        raise SystemExit("--zigzag-keep 须在 1..64")

    if args.reference_jpeg.strip():
        lum_z, chr_z = load_qtables_zigzag_from_jpeg(Path(args.reference_jpeg.strip()))
    else:
        lum_z, chr_z = default_qtables_zigzag()

    if args.input:
        src = Path(args.input)
        if not src.is_file():
            raise SystemExit(f"文件不存在: {src}")
        out = Path(args.out.strip()) if args.out.strip() else src.with_name(src.stem + args.suffix + ".jpg")
        written = process_one(src, out, args.tw, args.th, lum_z, chr_z, args.quality, zk)
        sz = written.stat().st_size
        print(f"已保存: {written.resolve()}  ({sz // 1024} KB), zigzag_keep={zk}")
        return

    folder = Path(args.dir)
    if not folder.is_dir():
        raise SystemExit(f"不是目录: {folder}")
    out_dir = Path(args.out_dir.strip()) if args.out_dir.strip() else folder / "wechat_jpg_hf_zero"
    n = 0
    for src in iter_image_paths(folder):
        out_path = default_out_path(src, out_dir, args.suffix)
        try:
            w = process_one(src, out_path, args.tw, args.th, lum_z, chr_z, args.quality, zk)
            print(f"{w.resolve()}  ({w.stat().st_size // 1024} KB)")
            n += 1
        except OSError as e:
            print(f"跳过 {src}: {e}")
    print(f"完成，共 {n} 张，zigzag_keep={zk}")


if __name__ == "__main__":
    main()
