"""Extract face crops from a dataset using a YOLO detector.

For every image under `<src>/.../<class>/`, runs detection and saves the
selected bbox crop (with optional margin) into the mirrored path under `<dst>`.
Images without any detection above the confidence threshold are logged to
`<dst>/_no_detection_report.txt` and skipped (the downstream pipeline expects
faces, not full frames — copying them would dilute the train signal).

Supports two source layouts:
  1) Flat:  <src>/<class>/<img.jpg>
  2) Split: <src>/{train,val,test}/<class>/<img.jpg>

Usage:
    uv run python scripts/extract_face_crops.py \\
        --src ./datasets/ceia_motion_detection_clean_split \\
        --dst ./datasets/ceia_motion_detection_clean_split_faces \\
        --model ./models/base/yolo26l.pt \\
        --margin 0.15
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from PIL import Image
from tqdm import tqdm

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_SPLIT_NAMES = ("train", "val", "test")


def _list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _VALID_EXTS)


def _select_box(result, strategy: str, target_class_ids: set[int] | None):
    """Return (x1, y1, x2, y2, conf) for the chosen detection, or None.

    `target_class_ids`, if provided, restricts which class ids count — handy
    when running a COCO-pretrained detector and only wanting the `cow` boxes,
    for example. Pass None to accept all classes.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    keep = list(range(len(boxes)))
    if target_class_ids is not None and boxes.cls is not None:
        cls_ids = [int(c) for c in boxes.cls.tolist()]
        keep = [i for i in keep if cls_ids[i] in target_class_ids]
        if not keep:
            return None
    if strategy == "area":
        xyxy = boxes.xyxy
        areas = [(float(xyxy[i, 2] - xyxy[i, 0]) * float(xyxy[i, 3] - xyxy[i, 1])) for i in keep]
        idx = keep[int(max(range(len(keep)), key=lambda j: areas[j]))]
    else:  # confidence
        confs = [float(boxes.conf[i].item()) for i in keep]
        idx = keep[int(max(range(len(keep)), key=lambda j: confs[j]))]
    x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[idx].tolist())
    conf = float(boxes.conf[idx].item())
    return x1, y1, x2, y2, conf


def _crop_with_margin(
    img: Image.Image, x1: float, y1: float, x2: float, y2: float, margin: float
) -> Image.Image:
    """Expand the bbox by `margin` fraction on each side, clamped to image bounds."""
    w = x2 - x1
    h = y2 - y1
    mx = w * margin
    my = h * margin
    x1c = max(0, int(round(x1 - mx)))
    y1c = max(0, int(round(y1 - my)))
    x2c = min(img.width, int(round(x2 + mx)))
    y2c = min(img.height, int(round(y2 + my)))
    return img.crop((x1c, y1c, x2c, y2c))


def _discover_image_dirs(src: Path) -> Iterator[tuple[str, Path]]:
    """Yield (relative_subpath, class_dir) for both flat and split layouts.

    Auto-detects: if `src` has subdirs named `train`/`val`/`test`, treats it
    as a split layout (`<src>/<split>/<class>/`); otherwise as flat
    (`<src>/<class>/`).
    """
    splits = sorted(d for d in src.iterdir() if d.is_dir() and d.name in _SPLIT_NAMES)
    if splits:
        for split_dir in splits:
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                yield f"{split_dir.name}/{class_dir.name}", class_dir
    else:
        for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
            yield class_dir.name, class_dir


def extract_crops(
    src: Path,
    dst: Path,
    model_path: Path,
    margin: float,
    confidence: float,
    select: str,
    image_size: int,
    device: str,
    batch_size: int,
    target_classes: list[str] | None,
) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if dst.exists():
        raise FileExistsError(f"{dst} already exists. Remove it or pick a new --dst.")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "extract_face_crops.py needs `ultralytics`. Run `uv sync` after "
            "the pyproject.toml update."
        ) from e

    dst.mkdir(parents=True)
    model = YOLO(str(model_path))

    # Resolve --target-classes (names or numeric ids) against model.names.
    target_class_ids: set[int] | None = None
    if target_classes:
        name_to_id = {v: k for k, v in model.names.items()}
        target_class_ids = set()
        for t in target_classes:
            if t.isdigit():
                target_class_ids.add(int(t))
            elif t in name_to_id:
                target_class_ids.add(int(name_to_id[t]))
            else:
                raise ValueError(
                    f"--target-classes value {t!r} not found in model.names "
                    f"({sorted(name_to_id)[:10]}...)"
                )
        print(f"Filtering to class ids: {sorted(target_class_ids)}")

    n_total = 0
    n_cropped = 0
    no_detection: list[str] = []

    image_dirs = list(_discover_image_dirs(src))
    for rel_path, class_dir in tqdm(image_dirs, desc="Classes"):
        images = _list_images(class_dir)
        if not images:
            continue
        out_dir = dst / rel_path
        out_dir.mkdir(parents=True, exist_ok=True)

        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            results = model.predict(
                [str(p) for p in batch],
                conf=confidence,
                imgsz=image_size,
                device=device,
                verbose=False,
            )
            for img_path, result in zip(batch, results, strict=True):
                n_total += 1
                box = _select_box(result, select, target_class_ids)
                if box is None:
                    no_detection.append(str(img_path.relative_to(src)))
                    continue
                x1, y1, x2, y2, _ = box
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    crop = _crop_with_margin(img, x1, y1, x2, y2, margin)
                crop.save(out_dir / img_path.name, quality=95)
                n_cropped += 1

    report_path = dst / "_no_detection_report.txt"
    if no_detection:
        report_path.write_text(
            "Images with no detection above conf threshold "
            f"{confidence} (skipped — not present in <dst>):\n\n"
            + "\n".join(no_detection)
            + "\n"
        )

    print()
    print(f"Crops written to {dst}")
    print(f"  total processed : {n_total}")
    print(f"  crops emitted   : {n_cropped}")
    print(f"  no detection    : {len(no_detection)}")
    if no_detection:
        print(f"  detection rate  : {100 * n_cropped / max(n_total, 1):.2f}%")
        print(f"  see             : {report_path}")
    else:
        print("  detection rate  : 100.00%")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True,
                   help="Dataset root (flat <class>/ or split train/val/test/<class>/)")
    p.add_argument("--dst", type=Path, required=True,
                   help="Output root — created fresh; must not exist")
    p.add_argument("--model", type=Path, required=True, help="YOLO .pt checkpoint")
    p.add_argument("--margin", type=float, default=0.1,
                   help="Fractional padding around bbox on each side (default 0.1)")
    p.add_argument("--confidence", type=float, default=0.25,
                   help="Min detection confidence (default 0.25)")
    p.add_argument("--select", choices=("confidence", "area"), default="confidence",
                   help="With multiple boxes per image, pick by `confidence` or `area` (default confidence)")
    p.add_argument("--image-size", type=int, default=640,
                   help="YOLO inference image size — short side (default 640)")
    p.add_argument("--device", default="cuda", help="cuda or cpu (default cuda)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Inference batch size (default 16; lower if VRAM is tight)")
    p.add_argument("--target-classes", nargs="+", default=None,
                   help="Restrict detections to these class names or numeric ids "
                        "(default: accept all). Useful with COCO-pretrained models — "
                        "e.g. `--target-classes cow` to keep only `cow` boxes (id 19).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_crops(
        args.src, args.dst, args.model,
        args.margin, args.confidence, args.select,
        args.image_size, args.device, args.batch_size,
        args.target_classes,
    )
