import random
from pathlib import Path

import numpy as np


def cv2_imread(path: str):
    import cv2

    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def cv2_imwrite(path: str, image) -> bool:
    import cv2

    suffix = Path(path).suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(path)
    return True


def order_points_tl_tr_br_bl(points):
    pts = np.asarray(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(d)]
    bl = pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_qr_markers(model, image_path: str, conf: float = 0.6):
    results = model(image_path, verbose=False, conf=conf)
    if not results:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes.xywh) < 4:
        return []

    detections = []
    for box_xywh, box_xyxy, score in zip(result.boxes.xywh, result.boxes.xyxy, result.boxes.conf):
        cx, cy, _, _ = box_xywh.tolist()
        x1, y1, x2, y2 = box_xyxy.tolist()
        detections.append(
            {
                "center": [float(cx), float(cy)],
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score.item()) if hasattr(score, "item") else float(score),
            }
        )

    top4 = sorted(detections, key=lambda item: item["score"], reverse=True)[:4]
    ordered_centers = order_points_tl_tr_br_bl([item["center"] for item in top4])

    ordered = []
    remaining = top4.copy()
    for pt in ordered_centers:
        best_idx = min(
            range(len(remaining)),
            key=lambda i: (remaining[i]["center"][0] - pt[0]) ** 2 + (remaining[i]["center"][1] - pt[1]) ** 2,
        )
        ordered.append(remaining.pop(best_idx))
    return ordered


def build_global_dst_points(screen_w: int, screen_h: int, qr_size: int):
    half = qr_size / 2.0
    return np.array(
        [
            [half, half],
            [screen_w - half, half],
            [screen_w - half, screen_h - half],
            [half, screen_h - half],
        ],
        dtype=np.float32,
    )


def rectify_photo(image_path: str, markers, screen_w: int = 1920, screen_h: int = 1080, qr_size: int = 120):
    import cv2

    image = cv2_imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if len(markers) != 4:
        raise ValueError(f"Need 4 markers, got {len(markers)}")

    src_points = np.array([marker["center"] for marker in markers], dtype=np.float32)
    dst_points = build_global_dst_points(screen_w, screen_h, qr_size)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(image, matrix, (screen_w, screen_h))
    return warped, src_points, dst_points, matrix


def draw_debug_markers(image_path: str, markers, src_points=None):
    import cv2

    image = cv2_imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    for idx, marker in enumerate(markers):
        x1, y1, x2, y2 = map(int, marker["bbox"])
        cx, cy = map(int, marker["center"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(image, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"{idx}:{marker['score']:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if src_points is not None:
        labels = ["TL", "TR", "BR", "BL"]
        for label, pt in zip(labels, src_points):
            cv2.putText(
                image,
                label,
                (int(pt[0]) + 10, int(pt[1]) + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
    return image


def sample_random_crops(image, crop_size: int, count: int, rng: random.Random):
    h, w = image.shape[:2]
    if crop_size > h or crop_size > w:
        return []

    crops = []
    for idx in range(count):
        top = rng.randint(0, h - crop_size)
        left = rng.randint(0, w - crop_size)
        crop = image[top : top + crop_size, left : left + crop_size].copy()
        crops.append(
            {
                "index": idx,
                "crop_size": crop_size,
                "top": top,
                "left": left,
                "image": crop,
            }
        )
    return crops
