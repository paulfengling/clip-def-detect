"""Single-image and folder inference, also shared by the Gradio application."""
import argparse
import json
from pathlib import Path
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from forward_utils import calculate_similarity_map, get_adapted_single_class_text_embedding
from model.adapter import AdaptedCLIP
from model.clip import create_model
from model.runtime import resolve_device
from defect_detection import detect_defects, draw_detections
from utils import setup_seed

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def find_image_checkpoint(save_path):
    folder = Path(save_path)
    latest = folder / "image_adapter.pth"
    if latest.is_file():
        return latest
    candidates = [p for p in folder.glob("image_adapter_*.pth")
                  if p.stem.rsplit("_", 1)[-1].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No trained image adapter in {folder}")
    return max(candidates, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))


class Predictor:
    def __init__(self, save_path=ROOT / "ckpt/baseline", img_size=None, device="auto"):
        torch.set_num_threads(4)
        self.device = resolve_device(device)
        setup_seed(111, device=self.device)
        folder = Path(save_path)
        config_file = folder / "config.json"
        self.config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
        self.img_size = img_size or self.config.get("img_size", 518)
        if self.img_size < 56 or self.img_size % 14:
            raise ValueError("img_size must be a multiple of 14, at least 56")
        self.checkpoint = find_image_checkpoint(folder)
        clip_model = create_model(
            model_name=self.config.get("model_name", "ViT-L-14-336"),
            img_size=self.img_size, device=self.device, pretrained="openai", require_pretrained=True,
            force_quick_gelu=self.config.get("quick_gelu", False),
        )
        settings = {key: self.config.get(key, default) for key, default in {
            "text_adapt_weight": 0.1, "image_adapt_weight": 0.1,
            "text_adapt_until": 3, "image_adapt_until": 6, "relu": False,
        }.items()}
        self.model = AdaptedCLIP(clip_model, **settings).to(self.device).eval()
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.model.image_adapter.load_state_dict(state["image_adapter"])
        text_file = folder / "text_adapter.pth"
        self.adapt_text = text_file.is_file() and self.config.get("text_epoch", 1) > 0
        if self.adapt_text:
            state = torch.load(text_file, map_location="cpu", weights_only=True)
            self.model.text_adapter.load_state_dict(state["text_adapter"])
        elif self.config.get("text_epoch", 0) > 0:
            raise FileNotFoundError(f"Missing trained text adapter: {text_file}")
        self.model.requires_grad_(False)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), Image.Resampling.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.text_cache = {}
        self.thresholds = {}
        self.pixel_thresholds = {}
        self.score_type = "pixel_max"
        threshold_file = folder / "thresholds.json"
        if threshold_file.exists():
            calibration = json.loads(threshold_file.read_text(encoding="utf-8"))
            matches = calibration.get("inference_signature") == self.inference_signature() if "inference_signature" in calibration else (
                calibration.get("checkpoint") == file_signature(self.checkpoint) and calibration.get("img_size") == self.img_size)
            if matches:
                self.thresholds = calibration["thresholds"]
                self.score_type = calibration.get("score_type", "image_score")
                self.pixel_thresholds = calibration.get("pixel_thresholds", {})
            else:
                warnings.warn("Ignoring stale thresholds.json; recalibrate after changing weights or inference settings.")

    def inference_signature(self):
        return {"image_adapter": file_signature(self.checkpoint),
                "text_adapter": file_signature(self.checkpoint.parent / "text_adapter.pth") if self.adapt_text else None,
                "img_size": self.img_size, "dataset": self.config.get("dataset", "MVTec"),
                **{key: self.config.get(key, default) for key, default in {
                    "model_name": "ViT-L-14-336", "quick_gelu": False,
                    "text_adapt_weight": 0.1, "image_adapt_weight": 0.1,
                    "text_adapt_until": 3, "image_adapt_until": 6, "relu": False}.items()}}

    @torch.inference_mode()
    def predict(self, image, class_name="bottle"):
        class_name = class_name.strip()
        if not class_name:
            raise ValueError("Class name must not be empty")
        if class_name not in self.text_cache:
            encoder = self.model if self.adapt_text else self.model.clipmodel
            self.text_cache[class_name] = get_adapted_single_class_text_embedding(
                encoder, self.config.get("dataset", "MVTec"), class_name, self.device,
            )
        text = self.text_cache[class_name]
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        features, detection = self.model(tensor)
        maps = [calculate_similarity_map(f, text, self.img_size, test=True, domain="Industrial") for f in features]
        anomaly_map = torch.cat(maps, dim=1).sum(1)[0].cpu().numpy()
        score = float(((detection @ text)[:, 1] + 1).item() / 2)
        if not np.isfinite(anomaly_map).all() or not np.isfinite(score):
            raise RuntimeError("Non-finite model prediction")
        return anomaly_map, score


def file_signature(path):
    path = Path(path)
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def inspect_prediction(predictor, image, anomaly_map, score, class_name,
                       threshold=None, score_type=None, pixel_threshold=None,
                       min_area=25, close_kernel=0, open_kernel=0, padding=0,
                       image_gate=True):
    """Shared classification/localization postprocessing for CLI and web UI."""
    class_name = class_name.strip()
    selected_score = score_type or predictor.score_type
    if selected_score not in ("image_score", "pixel_max"):
        raise ValueError("score_type must be image_score or pixel_max")
    record = {"class_name": class_name, "image_score": float(score),
              "pixel_max": float(np.max(anomaly_map)), "checkpoint": str(predictor.checkpoint),
              "img_size": predictor.img_size, "image_size": list(image.size)}
    if threshold is None and selected_score == predictor.score_type:
        threshold = predictor.thresholds.get(class_name)
    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError("Classification threshold must be finite")
        record.update(threshold=float(threshold), score_type=selected_score,
                      prediction="defect" if record[selected_score] >= threshold else "normal")
    else:
        record["prediction"] = "uncalibrated"
    if pixel_threshold is None:
        pixel_threshold = predictor.pixel_thresholds.get(class_name)
    record.update(detections=None, detection_status="uncalibrated",
                  box_format="xyxy; original pixels; right/bottom exclusive")
    mask = None
    boxed = image.convert("RGB").copy()
    if pixel_threshold is not None:
        mask, detections = detect_defects(anomaly_map, image.size, pixel_threshold,
                                         min_area=min_area, close_kernel=close_kernel,
                                         open_kernel=open_kernel, padding=padding)
        if image_gate and record["prediction"] == "normal":
            mask = np.zeros_like(mask)
            detections = []
            record["detection_status"] = "suppressed_by_image_threshold"
        else:
            record["detection_status"] = "ok"
        record.update(pixel_threshold=float(pixel_threshold), pixel_comparison=">",
                      detections=detections, num_defects=len(detections),
                      detection_settings={"min_area": min_area, "close_kernel": close_kernel,
                                          "open_kernel": open_kernel, "padding": padding,
                                          "image_gate": image_gate})
        boxed = draw_detections(image, detections)
    return boxed, Image.fromarray(mask) if mask is not None else None, record


def render_prediction(image, anomaly_map):
    resized = cv2.resize(anomaly_map, image.size, interpolation=cv2.INTER_LINEAR)
    span = max(float(np.ptp(resized)), 1e-8)
    gray = np.uint8(np.clip((resized - resized.min()) / span, 0, 1) * 255)
    heatmap = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.asarray(image.convert("RGB")), 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay), Image.fromarray(heatmap)


def main():
    parser = argparse.ArgumentParser(description="AA-CLIP defect inference")
    parser.add_argument("--image", type=Path, default=ROOT / "data/opencvxuetang/unknown/small_bad.png")
    parser.add_argument("--class_name", default="bottle")
    parser.add_argument("--save_path", type=Path, default=ROOT / "ckpt/baseline")
    parser.add_argument("--output", type=Path, default=ROOT / "results/predictions")
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--threshold", type=float, help="Optional threshold calibrated on validation data")
    parser.add_argument("--score_type", choices=["pixel_max", "image_score"], help="Score used with a manual threshold")
    parser.add_argument("--pixel_threshold", type=float, help="Raw pixel-response threshold for boxes; otherwise use validation calibration")
    parser.add_argument("--min_area", type=int, default=25, help="Minimum connected defect area in ORIGINAL image pixels")
    parser.add_argument("--close_kernel", type=int, default=0, help="Odd morphological closing size in original pixels; 0 disables")
    parser.add_argument("--open_kernel", type=int, default=0, help="Odd morphological opening size in original pixels; 0 disables")
    parser.add_argument("--padding", type=int, default=0, help="Box padding in original pixels")
    parser.add_argument("--no_image_gate", action="store_true", help="Keep pixel detections even when image classification is normal")
    args = parser.parse_args()
    files = sorted(p for p in args.image.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS) if args.image.is_dir() else [args.image]
    if not files or any(not p.is_file() for p in files):
        parser.error(f"No input images found: {args.image}")
    predictor = Predictor(args.save_path, args.img_size, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in files:
        with Image.open(path) as source:
            image = source.convert("RGB")
        anomaly_map, score = predictor.predict(image, args.class_name)
        overlay, heatmap = render_prediction(image, anomaly_map)
        boxed, mask, record = inspect_prediction(
            predictor, image, anomaly_map, score, args.class_name, args.threshold,
            args.score_type, args.pixel_threshold, args.min_area, args.close_kernel,
            args.open_kernel, args.padding, image_gate=not args.no_image_gate)
        relative = path.relative_to(args.image) if args.image.is_dir() else Path(path.name)
        prefix = args.output / relative.parent / relative.name
        prefix.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(str(prefix) + "_overlay.png")
        heatmap.save(str(prefix) + "_heatmap.png")
        boxed.save(str(prefix) + "_boxes.png")
        mask_path = Path(str(prefix) + "_mask.png")
        if mask is not None:
            mask.save(mask_path)
        else:
            mask_path.unlink(missing_ok=True)
        np.save(str(prefix) + "_raw.npy", anomaly_map)
        record["image"] = str(path)
        records.append(record)
        print(json.dumps(record, ensure_ascii=True), flush=True)
    (args.output / "predictions.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
