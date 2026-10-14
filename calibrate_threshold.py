"""Calibrate per-class anomaly thresholds on a held-out validation JSONL."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve

from custom_infer import Predictor
from data_preparation import audit_splits
from dataset import read_metadata
from defect_detection import calibrate_pixel_threshold, calibrate_pixel_dice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--meta_path", type=Path, required=True)
    parser.add_argument("--score_type", choices=["pixel_max", "image_score"], default="pixel_max")
    parser.add_argument("--pixel_fpr", type=float, default=0.001,
                        help="With --pixel_method normal_fpr: target false-positive pixel fraction on validation NORMAL images, before filtering")
    parser.add_argument("--pixel_method", choices=["dice", "normal_fpr"], default="dice",
                        help="Pixel threshold criterion: pooled validation-mask Dice (default), or validation-normal pixel FPR")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--img_size", type=int, help="Override training resolution and recalibrate for this resolution")
    args = parser.parse_args()
    if not 0 <= args.pixel_fpr < 1:
        parser.error("--pixel_fpr must be in [0, 1)")
    rows = read_metadata(args.data_root, args.meta_path)
    config_path = args.save_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if config.get("meta_path"):
        training_root = Path(config.get("data_root") or args.data_root)
        training = read_metadata(training_root, config["meta_path"])
        training_paths = {(training_root / row["image_path"]).resolve() for row in training}
        if any((args.data_root / row["image_path"]).resolve() in training_paths for row in rows):
            parser.error("Validation set overlaps the training set")
        # File renaming does not make a duplicate image independent evidence.
        seen = {entry["image_sha256"] for entry in audit_splits(training_root, {"train": training})}
        if any(entry["image_sha256"] in seen for entry in audit_splits(args.data_root, {"val": rows})):
            parser.error("Validation image content overlaps the training set")
    else:
        audit_splits(args.data_root, {"val": rows})
    for name in sorted({row["class_name"] for row in rows}):
        if {row["label"] for row in rows if row["class_name"] == name} != {0, 1}:
            parser.error(f"Validation needs normal AND defective images for {name}")
    predictor = Predictor(args.save_path, args.img_size, args.device)
    results = []
    normal_maps = {}
    normal_sizes = {}
    validation_maps = {}
    validation_masks = {}
    validation_sizes = {}
    for row in rows:
        with Image.open(args.data_root / row["image_path"]) as image:
            anomaly_map, score = predictor.predict(image, row["class_name"])
            if row["label"] == 0:
                normal_maps.setdefault(row["class_name"], []).append(anomaly_map)
                normal_sizes.setdefault(row["class_name"], []).append(image.size)
            if args.pixel_method == "dice":
                if row["label"] == 1:
                    with Image.open(args.data_root / row["mask_path"]) as mask:
                        target = np.array(mask) != 0
                else:
                    target = np.zeros((image.height, image.width), dtype=bool)
                validation_maps.setdefault(row["class_name"], []).append(anomaly_map)
                validation_masks.setdefault(row["class_name"], []).append(target)
                validation_sizes.setdefault(row["class_name"], []).append(image.size)
        results.append({"image_path": row["image_path"], "class_name": row["class_name"],
                        "label": row["label"], "image_score": score, "pixel_max": float(anomaly_map.max())})
    thresholds = {}
    for name in sorted({row["class_name"] for row in results}):
        samples = [row for row in results if row["class_name"] == name]
        labels = [row["label"] for row in samples]
        if set(labels) != {0, 1}:
            parser.error(f"Validation needs normal AND defective images for {name}")
        fpr, tpr, candidates = roc_curve(labels, [row[args.score_type] for row in samples])
        finite = np.isfinite(candidates)
        thresholds[name] = float(candidates[finite][np.argmax((tpr - fpr)[finite])])
    if args.pixel_method == "dice":
        pixel_calibration = {name: calibrate_pixel_dice(validation_maps[name], validation_masks[name],
                                                       image_sizes=validation_sizes[name]) for name in validation_maps}
    else:
        pixel_calibration = {name: calibrate_pixel_threshold(normal_maps[name], args.pixel_fpr,
                                                            image_sizes=normal_sizes[name]) for name in normal_maps}
    stat = predictor.checkpoint.stat()
    output = {"thresholds": thresholds, "img_size": predictor.img_size, "score_type": args.score_type,
              "checkpoint": {"name": predictor.checkpoint.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
              "criterion": "Maximum validation balanced accuracy (Youden J)",
              "validation_samples": results, "validation_meta": str(args.meta_path.resolve()),
              "inference_signature": predictor.inference_signature(),
              "pixel_thresholds": {name: entry["pixel_threshold"] for name, entry in pixel_calibration.items()},
              "pixel_method": args.pixel_method,
              "pixel_calibration": pixel_calibration}
    destination = args.save_path / "thresholds.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(f"Image thresholds: {thresholds}; pixel thresholds: {output['pixel_thresholds']}; saved to {destination}")


if __name__ == "__main__":
    main()
