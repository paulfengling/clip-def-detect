"""Validation and manifest export shared by the dataset preparation scripts."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def validate_record(root, row):
    image_path = Path(root) / row["image_path"]
    if row["label"] not in (0, 1):
        raise ValueError(f"Label must be 0 or 1: {image_path}")
    with Image.open(image_path) as source:
        rgb = source.convert("RGB")
        size = rgb.size
        digest = hashlib.sha256(str(size).encode("ascii") + rgb.tobytes()).hexdigest()
    coverage = 0.0
    if row["label"]:
        if not row.get("mask_path"):
            raise ValueError(f"Defective image has no mask: {image_path}")
        mask_path = Path(root) / row["mask_path"]
        with Image.open(mask_path) as mask:
            if mask.size != size:
                raise ValueError(f"Mask/image size mismatch: {mask_path}: {mask.size} != {size}")
            if mask.mode not in ("1", "L"):
                raise ValueError(f"Mask must be a grayscale binary PNG, not {mask.mode}: {mask_path}")
            values = np.asarray(mask)
            if not np.isin(values, [0, 1, 255]).all():
                raise ValueError(f"Mask contains non-binary values; use 0/255 PNG: {mask_path}")
            coverage = float(np.count_nonzero(values) / values.size)
            if coverage == 0:
                raise ValueError(f"Empty defect mask: {mask_path}")
    elif row.get("mask_path"):
        raise ValueError(f"Normal images should have an empty mask_path: {image_path}")
    return {"image_path": row["image_path"], "width": size[0], "height": size[1],
            "defect_fraction": coverage, "image_sha256": digest}


def audit_splits(root, splits):
    seen = {}
    audit = []
    for split, rows in splits.items():
        for row in rows:
            result = validate_record(root, row)
            fingerprint = result["image_sha256"]
            if fingerprint in seen:
                raise ValueError(f"Duplicate image content: {row['image_path']} ({split}) and {seen[fingerprint]}")
            seen[fingerprint] = f"{row['image_path']} ({split})"
            audit.append({"split": split, "class_name": row["class_name"], "label": row["label"], **result})
    return audit


def write_splits(root, splits, output, config, audit=None):
    audit = audit_splits(root, splits) if audit is None else audit
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        destination = output / f"{split}.jsonl"
        destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        counts = dict(Counter(f"{r['class_name']}:{'defect' if r['label'] else 'normal'}" for r in rows))
        print(f"{split}: {len(rows)} images; {counts}; {destination}")
    (output / "split_config.json").write_text(json.dumps({"data_root": str(Path(root).resolve()), **config},
                                                       indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
