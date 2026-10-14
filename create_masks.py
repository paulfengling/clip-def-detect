"""Convert Labelme polygons/rectangles into binary defect masks and overlays."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def convert_annotation(annotation, image_root, annotation_root, labels):
    annotation = Path(annotation)
    data = json.loads(annotation.read_text(encoding="utf-8-sig"))
    relative = annotation.relative_to(annotation_root)
    image_name = data.get("imagePath")
    if not image_name:
        raise ValueError(f"Missing imagePath: {annotation}")
    image_path = (Path(image_root) / relative.parent / image_name).resolve()
    if not image_path.is_relative_to(Path(image_root).resolve()):
        raise ValueError(f"imagePath must stay inside --images: {annotation}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    declared_size = (data.get("imageWidth"), data.get("imageHeight"))
    if declared_size != image.size:
        raise ValueError(f"Annotation/image size mismatch: {annotation}: {declared_size} != {image.size}")
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for shape in data.get("shapes", []):
        if shape.get("label") not in labels:
            raise ValueError(f"Unexpected label {shape.get('label')!r} in {annotation}; allowed: {sorted(labels)}")
        kind = shape.get("shape_type") or "polygon"
        points = np.asarray(shape.get("points", []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError(f"Invalid coordinates in {annotation}")
        if (points < 0).any() or (points[:, 0] > image.width).any() or (points[:, 1] > image.height).any():
            raise ValueError(f"Annotation points outside image in {annotation}")
        vertices = [tuple(point) for point in points]
        if kind == "polygon" and len(vertices) >= 3:
            area = abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))
            if area == 0:
                raise ValueError(f"Zero-area defect polygon in {annotation}")
            draw.polygon(vertices, fill=255)
        elif kind == "rectangle" and len(vertices) == 2:
            x0, y0 = points.min(axis=0)
            x1, y1 = points.max(axis=0)
            if x0 == x1 or y0 == y1:
                raise ValueError(f"Zero-area rectangle in {annotation}")
            draw.rectangle((x0, y0, x1, y1), fill=255)
        else:
            raise ValueError(f"Use polygons or rectangles with valid points, not {kind!r}: {annotation}")
    mask_array = np.asarray(mask)
    if not mask_array.any():
        raise ValueError(f"No annotated defect pixels in {annotation}")
    overlay = np.asarray(image).copy()
    foreground = mask_array > 0
    overlay[foreground] = (0.55 * overlay[foreground] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    return image_path, mask, Image.fromarray(overlay)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Folder of Labelme JSONs, searched recursively")
    parser.add_argument("--images", type=Path, help="Image tree matching the JSON tree; defaults to --annotations")
    parser.add_argument("--output", type=Path, required=True, help="Output mask root; previews go into _previews")
    parser.add_argument("--labels", nargs="+", default=["defect"])
    parser.add_argument("--overwrite", action="store_true", help="Replace existing masks/previews")
    args = parser.parse_args()
    image_root = args.images or args.annotations
    annotations = sorted(args.annotations.rglob("*.json"))
    if not annotations:
        parser.error(f"No Labelme JSON files in {args.annotations}")
    destinations = set()
    plans = []
    # Validate the complete batch before creating any outputs.
    for annotation in annotations:
        image_path, _, _ = convert_annotation(annotation, image_root, args.annotations, set(args.labels))
        relative_image = image_path.relative_to(image_root.resolve())
        mask_path = args.output / relative_image.parent / f"{relative_image.stem}_mask.png"
        preview_path = args.output / "_previews" / relative_image.parent / f"{relative_image.stem}_overlay.png"
        if mask_path in destinations:
            parser.error(f"Output name collision: {mask_path}; give images unique stems")
        destinations.add(mask_path)
        if not args.overwrite and (mask_path.exists() or preview_path.exists()):
            parser.error(f"Output exists: {mask_path}; use --overwrite to replace it")
        plans.append((annotation, mask_path, preview_path))
    for annotation, mask_path, preview_path in plans:
        _, mask, overlay = convert_annotation(annotation, image_root, args.annotations, set(args.labels))
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        mask.save(mask_path)
        overlay.save(preview_path)
        print(f"{annotation} -> {mask_path}")
    print(f"Created {len(plans)} masks. Check overlays in {args.output / '_previews'}")


if __name__ == "__main__":
    main()
