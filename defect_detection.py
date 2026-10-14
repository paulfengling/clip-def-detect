"""Turn raw AA-CLIP anomaly responses into defect regions and original-image boxes.

The model's anomaly maps are similarity responses, not probabilities. Thresholds
must therefore be supplied explicitly or calibrated for the same model, product
class, image preprocessing and response-map construction. Per-image min/max
normalization is useful for display, but must never be used here for decisions.
"""
from collections.abc import Iterable
from itertools import zip_longest
import math
from numbers import Integral, Real

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_number(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _image_size(image_size):
    try:
        width, height = image_size
    except (TypeError, ValueError) as exc:
        raise ValueError("image_size must be (width, height)") from exc
    return (_integer("image width", width, 1), _integer("image height", height, 1))


def _raw_map(anomaly_map):
    try:
        array = np.asarray(anomaly_map)
    except (TypeError, ValueError) as exc:
        raise ValueError("anomaly_map must be a finite 2D numeric array") from exc
    if (array.ndim != 2 or not array.size or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all()):
        raise ValueError("anomaly_map must be a nonempty finite 2D numeric array")
    # Float64 avoids a threshold rounding mismatch between calibration and use.
    return array.astype(np.float64, copy=False)


def resize_anomaly_map(anomaly_map, image_size):
    """Resize a raw response map to (width, height), without changing its scale."""
    array = _raw_map(anomaly_map)
    width, height = _image_size(image_size)
    if array.shape == (height, width):
        return array.copy()
    return cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)


def detect_defects(anomaly_map, image_size, pixel_threshold, *, min_area=25,
                   min_area_fraction=0.0, open_kernel=0, close_kernel=0,
                   padding=0, connectivity=8, max_regions=None):
    """Return an original-size uint8 mask (0/255) and JSON-ready defect records.

    A pixel is anomalous iff its raw resized response is strictly greater than
    ``pixel_threshold``. Morphology is optional and uses square kernels measured
    in original-image pixels; zero disables an operation. Opening precedes
    closing. Area filtering is applied after morphology. ``min_area_fraction``
    is relative to the entire original image, and is combined with ``min_area``
    by taking the larger requirement. The returned mask includes only retained
    regions (also after ``max_regions``), so it agrees with the detection list.

    ``bbox_xyxy`` follows the half-open convention [x1, y1, x2, y2], suitable
    for NumPy crops ``image[y1:y2, x1:x2]``. Padding affects boxes only. Region
    scores are raw maximum/mean responses over postprocessed component pixels,
    and area counts those pixels, including any added by closing. These scores
    are not calibrated confidence probabilities. Results sort by descending
    maximum response, then area, then coordinates for reproducibility.
    """
    threshold = _finite_number("pixel_threshold", pixel_threshold)
    width, height = _image_size(image_size)
    min_area = _integer("min_area", min_area, 1)
    min_area_fraction = _finite_number("min_area_fraction", min_area_fraction)
    if not 0 <= min_area_fraction <= 1:
        raise ValueError("min_area_fraction must be between 0 and 1")
    padding = _integer("padding", padding)
    connectivity = _integer("connectivity", connectivity, 1)
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if max_regions is not None:
        max_regions = _integer("max_regions", max_regions, 1)
    for name, kernel in (("open_kernel", open_kernel), ("close_kernel", close_kernel)):
        _integer(name, kernel)
        if kernel and kernel % 2 == 0:
            raise ValueError(f"{name} must be zero or a positive odd integer")
    response = resize_anomaly_map(anomaly_map, (width, height))
    binary = np.asarray(response > threshold, dtype=np.uint8)
    # Replicate the edge so closing does not erase defects touching a boundary.
    for operation, size in ((cv2.MORPH_OPEN, open_kernel), (cv2.MORPH_CLOSE, close_kernel)):
        if size:
            binary = cv2.morphologyEx(binary, operation, np.ones((size, size), np.uint8),
                                      borderType=cv2.BORDER_REPLICATE)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)
    required_area = max(min_area, math.ceil(min_area_fraction * width * height))
    regions = []
    for component in range(1, count):
        x, y, box_width, box_height, area = map(int, stats[component])
        if area < required_area:
            continue
        component_pixels = response[y:y + box_height, x:x + box_width][
            labels[y:y + box_height, x:x + box_width] == component]
        regions.append((component, {
            "bbox_xyxy": [max(0, x - padding), max(0, y - padding),
                          min(width, x + box_width + padding), min(height, y + box_height + padding)],
            "score_max": float(component_pixels.max()),
            "score_mean": float(component_pixels.mean()),
            "area": area,
            "area_fraction": area / (width * height),
        }))
    regions.sort(key=lambda item: (-item[1]["score_max"], -item[1]["area"], item[1]["bbox_xyxy"]))
    if max_regions is not None:
        regions = regions[:max_regions]
    kept_ids = [component for component, _ in regions]
    mask = np.isin(labels, kept_ids).astype(np.uint8) * 255
    return mask, [record for _, record in regions]


def draw_detections(image, detections, *, color=(255, 40, 40), line_width=2, show_scores=True):
    """Draw half-open boxes on a copy of a PIL image in its original dimensions."""
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL.Image.Image")
    line_width = _integer("line_width", line_width, 1)
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    for index, detection in enumerate(detections, start=1):
        box = detection["bbox_xyxy"]
        if len(box) != 4:
            raise ValueError("bbox_xyxy must have four coordinates")
        x1, y1, x2, y2 = [_integer("box coordinate", value) for value in box]
        if not 0 <= x1 < x2 <= image.width or not 0 <= y1 < y2 <= image.height:
            raise ValueError("bbox_xyxy must be nonempty and inside the image")
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=line_width)
        if show_scores:
            score = _finite_number("score_max", detection["score_max"])
            label = f"{index}: {score:.3f}"
            left, top, right, bottom = draw.textbbox((0, 0), label)
            label_width, label_height = right - left + 4, bottom - top + 4
            label_x = min(x1, max(0, image.width - label_width))
            label_y = max(0, y1 - label_height)
            draw.rectangle((label_x, label_y, label_x + label_width, label_y + label_height), fill=color)
            draw.text((label_x + 2 - left, label_y + 2 - top), label, fill=(255, 255, 255))
    return rendered


def calibrate_pixel_threshold(normal_maps: Iterable, false_positive_rate=0.001, *, image_sizes=None):
    """Choose a raw threshold limiting pooled *validation-normal pixel* FPR.

    Pass held-out normal images' response maps for one product class. If inference
    thresholds maps at the original image resolution, also pass each normal
    image's ``(width, height)``; calibration then applies exactly the same linear
    resize. Larger images contribute more pixels to this pooled distribution.
    Calibration without ``image_sizes`` uses the maps' own pixel grid.

    The largest permitted number of false-positive pixels is
    floor(false_positive_rate * number_of_pixels). An order statistic with a
    strict ``>`` decision achieves at most this count on the supplied maps,
    including tied values. The reported rate is measured before morphology and
    area filtering. It neither guarantees a future/test FPR nor controls the
    false-positive *image* rate. Use a representative independent validation set
    and recalibrate after changing weights, resolution, prompts or domain.

    Exact calibration stores all selected pixels in memory. A target of zero
    returns the maximum observed normal response. Values must be finite.
    """
    target = _finite_number("false_positive_rate", false_positive_rate)
    if not 0 <= target < 1:
        raise ValueError("false_positive_rate must be in [0, 1)")
    sizes = iter(image_sizes) if image_sizes is not None else None
    normal_pixels = []
    for raw in normal_maps:
        if sizes is None:
            response = _raw_map(raw)
        else:
            try:
                size = next(sizes)
            except StopIteration as exc:
                raise ValueError("image_sizes must match the number of normal_maps") from exc
            response = resize_anomaly_map(raw, size)
        normal_pixels.append(response.ravel())
    if not normal_pixels:
        raise ValueError("At least one normal validation response map is required")
    if sizes is not None:
        try:
            next(sizes)
        except StopIteration:
            pass
        else:
            raise ValueError("image_sizes must match the number of normal_maps")
    pixels = np.concatenate(normal_pixels)
    allowed = math.floor(target * pixels.size)
    index = pixels.size - allowed - 1
    threshold = float(np.partition(pixels, index)[index])
    false_positives = int(np.count_nonzero(pixels > threshold))
    return {
        "pixel_threshold": threshold,
        "comparison": ">",
        "criterion": "Pooled normal-validation pixel false-positive rate before postprocessing",
        "target_pixel_fpr": target,
        "observed_pixel_fpr": false_positives / int(pixels.size),
        "false_positive_pixels": false_positives,
        "normal_pixels": int(pixels.size),
        "normal_images": len(normal_pixels),
        "resolution": "original_image" if image_sizes is not None else "response_map",
    }


def _binary_validation_mask(mask):
    array = np.asarray(mask)
    if (array.ndim != 2 or not array.size or array.dtype.kind not in "bifu"
            or not np.isfinite(array).all() or not np.isin(array, (0, 1, 255)).all()):
        raise ValueError("Validation masks must be nonempty 2D binary arrays with values 0/1 or 0/255")
    return array != 0


def calibrate_pixel_dice(anomaly_maps: Iterable, masks: Iterable, *, image_sizes=None):
    """Maximize pooled validation pixel Dice using raw responses and binary masks.

    Maps and masks must describe the same held-out images from one product
    class. Masks must be original-image grayscale arrays with values 0/1 or
    0/255; at least one foreground pixel is required across the validation set.
    Each response map is resized with exactly ``resize_anomaly_map`` to its
    mask's original dimensions. Optional ``image_sizes`` is a matching iterable
    of (width, height) and must agree with the masks; masks are never resized.

    The optimization uses a single sort of all validation pixels. Equal-score
    groups are kept together, matching the strict ``>`` decision in detection.
    Candidate thresholds include every unique score and, when representable
    as a finite float, a threshold immediately below the smallest score. Equal
    maximum Dice selects the higher threshold, predicting fewer pixels.

    This is *pooled* Dice before morphology or area filtering: larger images
    contribute more pixels. It uses foreground and normal background pixels
    from both defective and normal validation images. Reported background FPR
    is None if no background pixels exist. This in-sample optimum is not a test
    accuracy guarantee and does not optimize box IoU or per-defect recall.
    Exact optimization holds and sorts all resized validation pixels in memory.
    """
    sizes = iter(image_sizes) if image_sizes is not None else None
    score_blocks, foreground_blocks = [], []
    sentinel = object()
    for raw, mask in zip_longest(anomaly_maps, masks, fillvalue=sentinel):
        if raw is sentinel or mask is sentinel:
            raise ValueError("masks must match the number of anomaly_maps")
        foreground = _binary_validation_mask(mask)
        height, width = foreground.shape
        if sizes is not None:
            try:
                image_size = _image_size(next(sizes))
            except StopIteration as exc:
                raise ValueError("image_sizes must match the number of anomaly_maps") from exc
            if image_size != (width, height):
                raise ValueError("Validation masks must match the original image_sizes")
        response = resize_anomaly_map(raw, (width, height))
        score_blocks.append(response.ravel())
        foreground_blocks.append(foreground.ravel())
    if not score_blocks:
        raise ValueError("At least one validation response map and mask is required")
    if sizes is not None:
        try:
            next(sizes)
        except StopIteration:
            pass
        else:
            raise ValueError("image_sizes must match the number of anomaly_maps")
    validation_images = len(score_blocks)
    scores = np.concatenate(score_blocks)
    foreground = np.concatenate(foreground_blocks)
    del score_blocks, foreground_blocks
    num_pixels = int(scores.size)
    num_foreground = int(np.count_nonzero(foreground))
    if not num_foreground:
        raise ValueError("Dice calibration requires foreground defect pixels in validation masks")
    order = np.argsort(scores)
    scores = scores[order]
    foreground = foreground[order]
    del order
    # A threshold equal to each unique score excludes the entire tied group.
    group_ends = np.flatnonzero(np.concatenate((scores[:-1] != scores[1:], [True])))
    cumulative_foreground = np.cumsum(foreground, dtype=np.int64)
    true_positives = num_foreground - cumulative_foreground[group_ends]
    del cumulative_foreground
    predicted_pixels = num_pixels - (group_ends + 1)
    dice = (2.0 * true_positives) / (num_foreground + predicted_pixels)
    # Reverse candidates so argmax resolves equal Dice toward higher threshold.
    best = len(dice) - 1 - int(np.argmax(dice[::-1]))
    threshold = float(scores[group_ends[best]])
    best_dice = float(dice[best])
    best_true_positives = int(true_positives[best])
    best_predicted_pixels = int(predicted_pixels[best])
    all_foreground_threshold = float(np.nextafter(scores[0], -np.inf))
    all_foreground_dice = 2.0 * num_foreground / (num_foreground + num_pixels)
    if math.isfinite(all_foreground_threshold) and all_foreground_dice > best_dice:
        threshold = all_foreground_threshold
        best_dice = all_foreground_dice
        best_true_positives = num_foreground
        best_predicted_pixels = num_pixels
    background_pixels = num_pixels - num_foreground
    false_positive_pixels = best_predicted_pixels - best_true_positives
    return {
        "pixel_threshold": threshold,
        "comparison": ">",
        "criterion": "Maximum pooled validation pixel Dice before postprocessing",
        "validation_dice": best_dice,
        "num_pixels": num_pixels,
        "num_foreground_pixels": num_foreground,
        "num_background_pixels": background_pixels,
        "validation_true_positive_pixels": best_true_positives,
        "validation_false_positive_pixels": false_positive_pixels,
        "validation_predicted_pixels": best_predicted_pixels,
        "observed_normal_pixel_fpr": false_positive_pixels / background_pixels if background_pixels else None,
        "validation_images": validation_images,
        "resolution": "original_image",
    }
