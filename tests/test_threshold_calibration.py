import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import calibrate_threshold


class FakePredictor:
    def __init__(self, save_path, img_size=None, device="auto"):
        self.checkpoint = Path(save_path) / "image_adapter.pth"
        self.img_size = img_size or 56

    def predict(self, image, class_name):
        defective = np.asarray(image)[0, 0, 0] > 50
        response = np.full((4, 4), -3.0, dtype=np.float32)
        if defective:
            response[1:3, 1:3] = 2.0
        return response, 0.9 if defective else 0.1

    def inference_signature(self):
        return {"test_signature": 1, "img_size": self.img_size}


class ThresholdCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "ckpt"
        self.model.mkdir()
        (self.model / "image_adapter.pth").write_bytes(b"fake adapter")
        self.normal = {"image_path": "normal.png", "class_name": "wafer", "label": 0, "mask_path": ""}
        self.defect = {"image_path": "defect.png", "class_name": "wafer", "label": 1, "mask_path": "mask.png"}
        Image.new("RGB", (8, 8), (20, 30, 40)).save(self.root / "normal.png")
        Image.new("RGB", (8, 8), (90, 30, 40)).save(self.root / "defect.png")
        mask = np.zeros((8, 8), np.uint8)
        mask[2:6, 2:6] = 255
        Image.fromarray(mask).save(self.root / "mask.png")
        self.meta = self.root / "val.jsonl"
        self.write_rows([self.normal, self.defect])

    def write_rows(self, rows):
        self.meta.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def run_main(self, extra=()):
        argv = ["calibrate_threshold.py", "--save_path", str(self.model), "--data_root", str(self.root),
                "--meta_path", str(self.meta), *extra]
        with patch("sys.argv", argv), patch.object(calibrate_threshold, "Predictor", FakePredictor), contextlib.redirect_stdout(io.StringIO()):
            calibrate_threshold.main()
        return json.loads((self.model / "thresholds.json").read_text())

    def test_default_dice_and_classification_threshold_saved_separately(self):
        result = self.run_main()
        self.assertEqual(result["pixel_method"], "dice")
        self.assertEqual(result["thresholds"], {"wafer": 2.0})
        self.assertEqual(result["inference_signature"]["test_signature"], 1)
        self.assertEqual(result["pixel_calibration"]["wafer"]["comparison"], ">")
        self.assertTrue(np.isfinite(result["pixel_thresholds"]["wafer"]))
        self.assertFalse((self.model / "thresholds.json.tmp").exists())

    def test_normal_fpr_uses_only_normal_images_and_original_resolution(self):
        result = self.run_main(["--pixel_method", "normal_fpr", "--pixel_fpr", "0"])
        entry = result["pixel_calibration"]["wafer"]
        self.assertEqual(result["pixel_thresholds"], {"wafer": -3.0})
        self.assertEqual(entry["normal_images"], 1)
        self.assertEqual(entry["normal_pixels"], 64)
        self.assertEqual(entry["observed_pixel_fpr"], 0)

    def test_validation_requires_both_labels_before_model_loading(self):
        self.write_rows([self.normal])
        with self.assertRaises(SystemExit):
            self.run_main()

    def test_renamed_training_image_cannot_leak_into_validation(self):
        train_meta = self.root / "train.jsonl"
        with Image.open(self.root / "normal.png") as source:
            source.save(self.root / "renamed_normal.png")
        train_meta.write_text(json.dumps({**self.normal, "image_path": "renamed_normal.png"}) + "\n")
        (self.model / "config.json").write_text(json.dumps({"meta_path": str(train_meta), "data_root": str(self.root)}))
        with self.assertRaises(SystemExit):
            self.run_main()


if __name__ == "__main__":
    unittest.main()
