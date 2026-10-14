"""Export selected SAM3 state['masks'] to an AA-CLIP binary mask."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def save_sam3_mask(masks, image_path, output, indices=None, overwrite=False):
    """Accept binary HxW, NxHxW or Nx1xHxW NumPy arrays / torch tensors."""
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3 or masks.shape[0] == 0:
        raise ValueError("Expected nonempty HxW, NxHxW or Nx1xHxW SAM3 masks")
    if not np.isin(masks, [0, 1, 255]).all():
        raise ValueError("Use SAM3 state['masks'] (binary), not logits, probabilities or a colored overlay")
    if indices is None:
        if len(masks) != 1:
            raise ValueError("Multiple candidates: inspect them and explicitly select --indices")
        indices = [0]
    if not indices or any(i < 0 or i >= len(masks) for i in indices):
        raise ValueError(f"Select one or more mask indices in [0, {len(masks) - 1}]")
    combined = (masks[indices] != 0).any(axis=0)
    if not combined.any():
        raise ValueError("Selected masks have no defect pixels; review the annotation")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if combined.shape != (image.height, image.width):
        raise ValueError("Mask must use ORIGINAL image coordinates and size; restore cropped/resized predictions first")
    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("Save the binary mask as .png")
    preview_path = output.parent / "_previews" / f"{output.stem}_overlay.png"
    if not overwrite and (output.exists() or preview_path.exists()):
        raise FileExistsError(f"Output exists: {output}; pass overwrite=True or --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined.astype(np.uint8) * 255).save(output)
    overlay = np.asarray(image).copy()
    overlay[combined] = (overlay[combined] * 0.55 + np.array([255, 0, 0]) * 0.45).astype(np.uint8)
    Image.fromarray(overlay).save(preview_path)
    return output, preview_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True, help=".npy file containing binary SAM3 masks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = save_sam3_mask(np.load(args.masks, allow_pickle=False), args.image, args.output, args.indices, args.overwrite)
    print(f"Mask: {paths[0]}\nReview overlay: {paths[1]}")


if __name__ == "__main__":
    main()
