"""
带投票机制的照片水印解码

对同一张图片使用多个裁剪位置和大小的结果做投票/聚类，得到最终解码结果再计算准确率。

用法:
  python decode_photo_voting.py --photo-root test_photo/v0_30_128 --display-test-dir display_test2
  python decode_photo_voting.py --photo-root test_photo/v0_30_128 --vote-methods logit_mean,majority_vote
"""

import argparse
import importlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from decode_photo import (
    load_yaml_args,
    pil_to_tensor,
    message_to_tensor,
    decode_crops_batch,
    msg_acc_batch,
    metric_stats,
    resolve_ckpt,
    load_seed_messages,
    list_seed_dirs,
    list_images,
    parse_crop_sizes,
    random_crop_tensor,
    write_csv,
)


# ── 投票策略 ──────────────────────────────────────────────


def vote_logit_mean(all_preds: torch.Tensor) -> Tuple[str, torch.Tensor]:
    """对所有裁剪的连续 pred 取平均再 round。"""
    mean_pred = all_preds.mean(dim=0, keepdim=True)  # (1, msg_len)
    bits = torch.round(mean_pred).clamp(0, 1)
    msg = "".join(str(int(x)) for x in bits.squeeze(0).cpu())
    return msg, mean_pred


def vote_majority(all_preds: torch.Tensor) -> Tuple[str, torch.Tensor]:
    """逐位多数投票。"""
    hard_bits = torch.round(all_preds).clamp(0, 1)  # (N, msg_len)
    vote = (hard_bits.sum(dim=0) >= (all_preds.shape[0] / 2.0)).float().unsqueeze(0)
    msg = "".join(str(int(x)) for x in vote.squeeze(0).cpu())
    return msg, vote


def vote_confidence_weighted(all_preds: torch.Tensor) -> Tuple[str, torch.Tensor]:
    """以 |pred - 0.5| 为置信度权重做加权平均再 round。"""
    conf = (all_preds - 0.5).abs()  # (N, msg_len)
    conf_sum = conf.sum(dim=0, keepdim=True).clamp(min=1e-8)
    weights = conf / conf_sum
    weighted = (all_preds * weights).sum(dim=0, keepdim=True)
    bits = torch.round(weighted).clamp(0, 1)
    msg = "".join(str(int(x)) for x in bits.squeeze(0).cpu())
    return msg, weighted


def vote_size_weighted(all_preds: torch.Tensor, sizes: List[int], num_per_size: int) -> Tuple[str, torch.Tensor]:
    """按裁剪尺寸大小加权（大裁剪包含更多上下文）。"""
    weights = []
    for s in sizes:
        weights.extend([float(s)] * num_per_size)
    w = torch.tensor(weights, dtype=torch.float32, device=all_preds.device).unsqueeze(1)
    w = w / w.sum()
    weighted = (all_preds * w).sum(dim=0, keepdim=True)
    bits = torch.round(weighted).clamp(0, 1)
    msg = "".join(str(int(x)) for x in bits.squeeze(0).cpu())
    return msg, weighted


def vote_cluster(all_preds: torch.Tensor, msg_len: int) -> Tuple[str, torch.Tensor]:
    """按 bit 字符串聚类，取出现次数最多的结果。"""
    hard_bits = torch.round(all_preds).clamp(0, 1)
    msgs = ["".join(str(int(x)) for x in hard_bits[i].cpu()[:msg_len]) for i in range(hard_bits.shape[0])]
    counter = Counter(msgs)
    best_msg = counter.most_common(1)[0][0]
    best_tensor = torch.tensor(
        [float(c) for c in best_msg], dtype=torch.float32, device=all_preds.device
    ).unsqueeze(0)
    return best_msg, best_tensor


VOTE_FUNCS = {
    "logit_mean": vote_logit_mean,
    "majority_vote": vote_majority,
    "confidence_weighted": vote_confidence_weighted,
    "size_weighted": vote_size_weighted,
    "cluster": vote_cluster,
}


def apply_vote(
    method: str,
    all_preds: torch.Tensor,
    sizes: List[int],
    num_per_size: int,
    msg_len: int,
) -> Tuple[str, torch.Tensor]:
    """应用指定投票方法。size_weighted 在单尺寸时退化为 logit_mean。"""
    if method == "size_weighted" and len(sizes) == 1:
        return vote_logit_mean(all_preds)
    if method == "size_weighted":
        return vote_size_weighted(all_preds, sizes, num_per_size)
    if method == "cluster":
        return vote_cluster(all_preds, msg_len)
    return VOTE_FUNCS[method](all_preds)


def compute_voted_acc(voted_msg: str, target_msg: str, msg_len: int) -> float:
    """计算投票结果与目标消息的比特准确率。"""
    correct = sum(1 for a, b in zip(voted_msg, target_msg) if a == b)
    return correct / msg_len


# ── 主解码逻辑 ────────────────────────────────────────────


def decode_batch_voting(cli, model, msg_len: int, decode_size: int) -> None:
    seed_messages = load_seed_messages(cli.display_test_dir)
    if not seed_messages:
        raise FileNotFoundError(f"No seed messages found in {cli.display_test_dir}")

    crop_sizes = parse_crop_sizes(cli.crop_sizes)
    vote_methods = [m.strip() for m in cli.vote_methods.split(",") if m.strip()]
    for m in vote_methods:
        if m not in VOTE_FUNCS:
            raise ValueError(f"Unknown vote method: {m}. Available: {list(VOTE_FUNCS.keys())}")

    result_dir = Path(cli.result_dir)
    seed_dirs = list_seed_dirs(cli.photo_root)

    crop_rows: List[Dict] = []
    image_summary_rows: List[Dict] = []
    seed_summary_rows: List[Dict] = []
    size_scores: Dict[int, List[float]] = {size: [] for size in crop_sizes}
    total_image_count = 0

    # 投票统计：per-method → acc 值列表
    voting_mixed_scores: Dict[str, List[float]] = {m: [] for m in vote_methods}
    voting_size_scores: Dict[Tuple[str, int], List[float]] = {
        (m, s): [] for m in vote_methods for s in crop_sizes
    }

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

        # seed 级投票统计
        seed_voting_mixed: Dict[str, List[float]] = {m: [] for m in vote_methods}
        seed_voting_size: Dict[Tuple[str, int], List[float]] = {
            (m, s): [] for m in vote_methods for s in crop_sizes
        }

        print(f"[seed {seed_name}] images={len(image_paths)} target={target_msg}")

        for image_index, image_path in enumerate(image_paths):
            img = Image.open(image_path).convert("RGB")
            image_scores: List[float] = []
            image_size_scores: Dict[int, List[float]] = {size: [] for size in crop_sizes}

            # 收集该图片所有裁剪的 pred
            image_all_preds: List[torch.Tensor] = []
            image_size_preds: Dict[int, List[torch.Tensor]] = {size: [] for size in crop_sizes}

            img_processed = img

            if cli.pre_resize_w > 0 and cli.pre_resize_h > 0:
                img_processed = img_processed.resize((cli.pre_resize_w, cli.pre_resize_h), Image.BICUBIC)

            img_tensor = pil_to_tensor(img_processed)

            for crop_size in crop_sizes:
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

                for loop_idx, ((crop_index, crop_meta), decoded_msg, acc) in enumerate(zip(batch_metas, decoded_msgs, accs)):
                    acc_val = float(acc.item())
                    seed_scores.append(acc_val)
                    seed_size_scores[crop_size].append(acc_val)
                    image_scores.append(acc_val)
                    image_size_scores[crop_size].append(acc_val)
                    size_scores[crop_size].append(acc_val)

                    # 收集 pred 用于投票（detach 避免梯度累积）
                    image_all_preds.append(preds[loop_idx : loop_idx + 1].detach())
                    image_size_preds[crop_size].append(preds[loop_idx : loop_idx + 1].detach())

                    crop_rows.append({
                        "seed": seed_name,
                        "image_path": str(image_path),
                        "image_name": image_path.name,
                        "image_index": image_index,
                        "crop_size": crop_meta["size"],
                        "crop_index": crop_index,
                        "top": crop_meta["top"],
                        "left": crop_meta["left"],
                        "bit_acc": acc_val,
                        "decoded_msg": decoded_msg,
                        "target_msg": target_msg,
                    })

            # ── 对该图片应用投票 ──
            image_row = {
                "seed": seed_name,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "image_index": image_index,
                "crop_count": len(image_scores),
                "bit_acc_mean": metric_stats(image_scores)["mean"],
                "bit_acc_var": metric_stats(image_scores)["var"],
                "bit_acc_std": metric_stats(image_scores)["std"],
                "bit_acc_min": metric_stats(image_scores)["min"],
                "bit_acc_max": metric_stats(image_scores)["max"],
            }
            for crop_size in crop_sizes:
                image_row[f"bit_acc_{crop_size}_mean"] = metric_stats(image_size_scores[crop_size])["mean"]

            # 混合投票（所有尺寸）
            if len(image_all_preds) > 1:
                all_preds_tensor = torch.cat(image_all_preds, dim=0)
                for method in vote_methods:
                    voted_msg, _ = apply_vote(method, all_preds_tensor, crop_sizes, cli.num_crops_per_size, msg_len)
                    voted_acc = compute_voted_acc(voted_msg, target_msg, msg_len)
                    image_row[f"voted_{method}_mixed_acc"] = voted_acc
                    voting_mixed_scores[method].append(voted_acc)
                    seed_voting_mixed[method].append(voted_acc)

            # 分尺寸投票
            for size in crop_sizes:
                preds_for_size = image_size_preds[size]
                if len(preds_for_size) > 1:
                    size_preds_tensor = torch.cat(preds_for_size, dim=0)
                    for method in vote_methods:
                        voted_msg, _ = apply_vote(method, size_preds_tensor, [size], len(preds_for_size), msg_len)
                        voted_acc = compute_voted_acc(voted_msg, target_msg, msg_len)
                        image_row[f"voted_{method}_size{size}_acc"] = voted_acc
                        voting_size_scores[(method, size)].append(voted_acc)
                        seed_voting_size[(method, size)].append(voted_acc)

            image_summary_rows.append(image_row)

        # ── seed 级汇总 ──
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

        # seed 级投票汇总
        for method in vote_methods:
            if seed_voting_mixed[method]:
                seed_row[f"voted_{method}_mixed_mean"] = metric_stats(seed_voting_mixed[method])["mean"]
            for size in crop_sizes:
                key = (method, size)
                if seed_voting_size[key]:
                    seed_row[f"voted_{method}_size{size}_mean"] = metric_stats(seed_voting_size[key])["mean"]

        seed_summary_rows.append(seed_row)

        size_text = "  ".join(
            f"mean@{s}={metric_stats(seed_size_scores[s])['mean']:.4f}" for s in crop_sizes
        )
        print(
            f"  -> crops={len(seed_scores)}  mean={seed_stats['mean']:.4f}  "
            f"min={seed_stats['min']:.4f}  max={seed_stats['max']:.4f}  {size_text}"
        )

    # ── 写入结果 ──
    write_csv(result_dir / "crops.csv", crop_rows)
    write_csv(result_dir / "image_summary.csv", image_summary_rows)
    write_csv(result_dir / "seed_summary.csv", seed_summary_rows)

    # summary.json
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
        "device": cli.device,
        "crop_sizes": crop_sizes,
        "num_crops_per_size": cli.num_crops_per_size,
        "vote_methods": vote_methods,
        "num_seeds": len(seed_summary_rows),
        "num_images": total_image_count,
        "num_crops": len(crop_rows),
        "overall": metric_stats(overall_scores),
        "by_crop_size": {str(s): metric_stats(size_scores[s]) for s in crop_sizes},
        "voting": {
            "mixed": {m: metric_stats(voting_mixed_scores[m]) for m in vote_methods},
            "per_size": {
                f"{m}_size{s}": metric_stats(voting_size_scores[(m, s)])
                for m in vote_methods for s in crop_sizes
            },
        },
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # ── 终端打印 ──
    print(f"\n{'=' * 80}")
    print(f"投票方法: {', '.join(vote_methods)}")
    print(f"投票粒度: 分尺寸 + 混合")
    print(f"{'=' * 80}")

    # 每个 seed 的原始准确率
    print("\n每个 seed 的原始逐裁剪准确率:")
    for row in seed_summary_rows:
        print(
            f"  seed={row['seed']:>2}  images={int(row['image_count']):>3}  crops={int(row['crop_count']):>4}  "
            f"mean={float(row['bit_acc_mean']):.4f}  min={float(row['bit_acc_min']):.4f}  "
            f"max={float(row['bit_acc_max']):.4f}"
        )

    # 投票结果汇总（混合）
    print("\n投票结果汇总 - 混合（所有尺寸）:")
    print(f"  {'方法':<25} {'count':>6} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    print(f"  {'-' * 65}")
    for method in vote_methods:
        stats = metric_stats(voting_mixed_scores[method])
        if stats["count"] > 0:
            print(
                f"  {method:<25} {stats['count']:>6} {stats['mean']:>8.4f} "
                f"{stats['std']:>8.4f} {stats['min']:>8.4f} {stats['max']:>8.4f}"
            )

    # 投票结果汇总（分尺寸）
    print("\n投票结果汇总 - 分尺寸:")
    print(f"  {'方法':<20} {'尺寸':>6} {'count':>6} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    print(f"  {'-' * 70}")
    for method in vote_methods:
        for size in crop_sizes:
            stats = metric_stats(voting_size_scores[(method, size)])
            if stats["count"] > 0:
                print(
                    f"  {method:<20} {size:>6} {stats['count']:>6} {stats['mean']:>8.4f} "
                    f"{stats['std']:>8.4f} {stats['min']:>8.4f} {stats['max']:>8.4f}"
                )

    # 最佳投票方法
    print("\n最佳投票方法（按混合 mean 排序）:")
    ranked = sorted(vote_methods, key=lambda m: metric_stats(voting_mixed_scores[m])["mean"], reverse=True)
    for i, method in enumerate(ranked):
        stats = metric_stats(voting_mixed_scores[method])
        marker = " *" if i == 0 else ""
        print(f"  {i+1}. {method:<25} mean={stats['mean']:.4f}{marker}")

    print(f"\n结果已保存到: {result_dir}")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="带投票机制的照片水印解码")
    parser.add_argument("--photo-root", default="test_photo/v2_64_256_dual")
    parser.add_argument("--display-test-dir", default="display_test3_256_64_dual")
    parser.add_argument("--cfg", default="/home/lpl2025/lpl/inrsteg-final_v1/config/v6_size256_msg64.yaml")
    parser.add_argument(
        "--ckpt",
        default="/home/lpl2025/lpl/inrsteg-final_v1/output1/dual_noise/ismark_v6_size256_msg64/lightning_logs/version_3/checkpoints/ckpt-epoch=04-val_loss=0.1313.ckpt",
    )
    parser.add_argument("--model-module", default="model.ismark_v6_30bit")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep-proxy-env", action="store_true")
    parser.add_argument("--decode-size", type=int, default=None)
    parser.add_argument("--crop-sizes", default="128,256,512,1024")
    parser.add_argument("--num-crops-per-size", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260512)
    parser.add_argument("--result-dir", default="./decode_photo_results_voting")
    parser.add_argument("--pre-resize-w", type=int, default=0)
    parser.add_argument("--pre-resize-h", type=int, default=0)
    parser.add_argument("--edge-margin", type=int, default=0)
    parser.add_argument(
        "--vote-methods",
        default="logit_mean,majority_vote,confidence_weighted,size_weighted,cluster",
        help="投票方法, 逗号分隔。可选: logit_mean,majority_vote,confidence_weighted,size_weighted,cluster",
    )
    cli = parser.parse_args()

    if not cli.keep_proxy_env:
        for k in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
            os.environ.pop(k, None)

    random.seed(cli.random_seed)
    np.random.seed(cli.random_seed)
    torch.manual_seed(cli.random_seed)

    ckpt_path = resolve_ckpt(cli.ckpt)
    cfg = load_yaml_args(cli.cfg)
    module = importlib.import_module(cli.model_module)
    model = module.INRMark.load_from_checkpoint(ckpt_path, args=cfg)
    model = model.to(cli.device).eval()

    msg_len = int(cfg.msg_len)
    decode_size = int(cli.decode_size or cfg.img_size)

    decode_batch_voting(cli, model, msg_len, decode_size)


if __name__ == "__main__":
    main()
