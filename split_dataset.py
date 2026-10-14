"""Split custom class/good, class/defect, class/masks folders for AA-CLIP."""
import argparse
from collections import defaultdict
from pathlib import Path
import random

from data_preparation import IMAGE_EXTENSIONS, audit_splits, write_splits


def collect_records(root, classes):
    root = Path(root)
    rows = []
    mask_owners = {}
    for name in classes:
        for label, folder in ((0, "good"), (1, "defect")):
            source = root / name / folder
            paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            if not paths:
                raise ValueError(f"No images in {source}")
            for path in paths:
                relative = path.relative_to(source)
                mask = root / name / "masks" / relative.parent / f"{path.stem}_mask.png"
                if label:
                    if mask in mask_owners:
                        raise ValueError(f"Images share a mask filename: {path} and {mask_owners[mask]}; rename duplicate stems")
                    mask_owners[mask] = path
                rows.append({"image_path": path.relative_to(root).as_posix(), "label": label,
                             "mask_path": mask.relative_to(root).as_posix() if label else "", "class_name": name})
    return rows


def interleave(rows, rng):
    groups = defaultdict(list)
    for row in rows:
        groups[str(Path(row["image_path"]).parent)].append(row)
    for group in groups.values():
        rng.shuffle(group)
    ordered = []
    while any(groups.values()):
        keys = sorted(groups)
        rng.shuffle(keys)
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop())
    return ordered


def split_records(rows, shot=8, val_per_label=4, ratios=None, seed=111):
    if ratios is None and (shot < 2 or shot % 2 or val_per_label < 1):
        raise ValueError("Use an even shot >= 2 and val_per_label >= 1")
    if ratios is not None and (len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-6):
        raise ValueError("Three positive train/val/test ratios must sum to 1")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["class_name"], row["label"])].append(row)
    if not groups:
        raise ValueError("Empty dataset")
    for name, _ in groups:
        if (name, 0) not in groups or (name, 1) not in groups:
            raise ValueError(f"Both normal and defective images are required for {name}")
    splits = {"train": [], "val": [], "test": []}
    for (name, label), group in sorted(groups.items()):
        ordered = interleave(group, rng)
        if ratios is None:
            train_count, val_count = shot // 2, val_per_label
        else:
            train_count, val_count = int(len(group) * ratios[0]), int(len(group) * ratios[1])
        if min(train_count, val_count, len(group) - train_count - val_count) < 1:
            raise ValueError(f"Not enough {name} label={label} images ({len(group)}) for nonempty train/val/test; reduce counts or add data")
        splits["train"].extend(ordered[:train_count])
        splits["val"].extend(ordered[train_count:train_count + val_count])
        splits["test"].extend(ordered[train_count + val_count:])
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", required=True, help="Product folder names (also used as CLIP class names)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--shot", type=int, help="TOTAL training images per class, half normal, half defective; default 8")
    mode.add_argument("--ratios", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--val_per_label", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = collect_records(args.data_root, args.classes)
    splits = split_records(rows, args.shot if args.shot is not None else 8, args.val_per_label, args.ratios, args.seed)
    audit = audit_splits(args.data_root, splits)
    write_splits(args.data_root, splits, args.output, {
        "layout": "custom", "classes": args.classes, "shot_per_class": None if args.ratios else (args.shot if args.shot is not None else 8),
        "ratios": args.ratios, "val_per_label": None if args.ratios else args.val_per_label,
        "seed": args.seed, "note": "Image-level split. Related views/frames must be grouped before splitting.",
    }, audit=audit)


if __name__ == "__main__":
    main()
