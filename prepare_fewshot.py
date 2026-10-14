"""Prepare a reproducible, disjoint supervised split from an MVTec-style dataset."""
import argparse
from collections import defaultdict
from pathlib import Path
import random
from data_preparation import write_splits

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def make_split(root, classes, shot=8, seed=111, val_per_label=4):
    if shot < 2 or shot % 2:
        raise ValueError("shot must be an even number >= 2 (half normal, half defective)")
    if val_per_label < 1:
        raise ValueError("val_per_label must be >= 1")
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for name in classes:
        def records(split, normal):
            result = []
            for path in sorted((root / name / split).glob("*/*")):
                if path.suffix.lower() not in EXTENSIONS or (path.parent.name == "good") != normal:
                    continue
                mask = root / name / "ground_truth" / path.parent.name / (path.stem + "_mask.png")
                if not normal and not mask.is_file():
                    raise FileNotFoundError(f"Missing defect mask: {mask}")
                result.append({"image_path": path.relative_to(root).as_posix(), "label": int(not normal),
                               "mask_path": "" if normal else mask.relative_to(root).as_posix(), "class_name": name})
            return result

        good_train = records("train", True)
        good_test = records("test", True)
        defects = records("test", False)
        rng.shuffle(good_train)
        rng.shuffle(good_test)
        # Interleave defect types so a small training set covers multiple defects.
        groups = defaultdict(list)
        for row in defects:
            groups[Path(row["image_path"]).parent.name].append(row)
        for values in groups.values():
            rng.shuffle(values)
        defects = []
        while any(groups.values()):
            keys = list(groups)
            rng.shuffle(keys)
            for key in keys:
                if groups[key]:
                    defects.append(groups[key].pop())
        count = shot // 2
        if len(good_train) < count or len(good_test) <= val_per_label or len(defects) <= count + val_per_label:
            raise ValueError(f"Not enough images for disjoint train/val/test splits: {name}")
        splits["train"].extend(good_train[:count] + defects[:count])
        splits["val"].extend(good_test[:val_per_label] + defects[count:count + val_per_label])
        splits["test"].extend(good_test[val_per_label:] + defects[count + val_per_label:])
    seen = set()
    for rows in splits.values():
        paths = {row["image_path"] for row in rows}
        if paths & seen or len(paths) != len(rows):
            raise ValueError("Overlapping or duplicate images found")
        seen.update(paths)
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("data/opencvxuetang/MVTecAD"))
    parser.add_argument("--classes", nargs="+", default=["bottle"])
    parser.add_argument("--shot", type=int, default=8)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--val_per_label", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("dataset/metadata/bottle_fewshot"))
    args = parser.parse_args()
    splits = make_split(args.data_root, args.classes, args.shot, args.seed, args.val_per_label)
    write_splits(args.data_root, splits, args.output, {
        "classes": args.classes, "shot_per_class": args.shot,
        "seed": args.seed, "val_per_label": args.val_per_label,
        "note": "Supervised transfer split: selected original test defects are excluded from val/test. Not the standard MVTec benchmark split.",
    })


if __name__ == "__main__":
    main()
