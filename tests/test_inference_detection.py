"""Exercise the user-facing detection contract without loading model weights."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import custom_infer
import web_infer_app


class InferenceDetectionTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (12, 8), (20, 30, 40))
        self.response = np.zeros((4, 6), dtype=np.float32)
        self.response[1:3, 2:4] = 2.0
        self.predictor = SimpleNamespace(
            checkpoint=Path("/fake/image_adapter.pth"), img_size=56,
            thresholds={"wafer": 0.5}, score_type="image_score",
            pixel_thresholds={"wafer": 1.0},
            predict=Mock(return_value=(self.response, 0.9)),
        )

    def inspect(self, **options):
        score = options.pop("score", 0.9)
        return custom_infer.inspect_prediction(
            self.predictor, self.image, self.response, score, " wafer ", min_area=1, **options)

    def test_calibrated_pixel_threshold_and_original_coordinate_boxes(self):
        boxes, mask, record = self.inspect()
        self.assertEqual(record["class_name"], "wafer")
        self.assertEqual(record["prediction"], "defect")
        self.assertEqual(record["detection_status"], "ok")
        self.assertEqual(record["pixel_threshold"], 1.0)
        self.assertEqual(record["pixel_comparison"], ">")
        self.assertEqual(record["image_size"], [12, 8])
        self.assertEqual(record["detections"][0]["bbox_xyxy"], [4, 2, 8, 6])
        self.assertEqual(record["num_defects"], 1)
        self.assertEqual(mask.size, self.image.size)
        self.assertEqual(boxes.size, self.image.size)
        self.assertEqual(np.count_nonzero(np.asarray(mask)), record["detections"][0]["area"])
        json.dumps(record)  # Includes only JSON-compatible values, no NumPy objects.

    def test_image_gate_suppresses_boxes_with_explicit_status(self):
        boxes, mask, record = self.inspect(score=0.2)
        self.assertEqual(record["prediction"], "normal")
        self.assertEqual(record["detection_status"], "suppressed_by_image_threshold")
        self.assertEqual(record["detections"], [])
        self.assertEqual(record["num_defects"], 0)
        self.assertFalse(np.asarray(mask).any())
        np.testing.assert_array_equal(np.asarray(boxes), np.asarray(self.image))
        _, kept_mask, kept = self.inspect(score=0.2, image_gate=False)
        self.assertEqual(kept["prediction"], "normal")
        self.assertEqual(kept["detection_status"], "ok")
        self.assertEqual(kept["num_defects"], 1)
        self.assertTrue(np.asarray(kept_mask).any())

    def test_uncalibrated_detection_is_distinct_from_zero_detections(self):
        self.predictor.pixel_thresholds = {}
        boxes, mask, record = self.inspect()
        self.assertEqual(record["prediction"], "defect")
        self.assertEqual(record["detection_status"], "uncalibrated")
        self.assertIsNone(record["detections"])
        self.assertIsNone(mask)
        self.assertNotIn("num_defects", record)
        np.testing.assert_array_equal(np.asarray(boxes), np.asarray(self.image))
        self.predictor.thresholds = {}
        _, _, fully_uncalibrated = self.inspect()
        self.assertEqual(fully_uncalibrated["prediction"], "uncalibrated")
        self.assertEqual(fully_uncalibrated["detection_status"], "uncalibrated")

    def test_score_type_switch_does_not_reuse_differently_calibrated_image_threshold(self):
        _, mask, record = self.inspect(score=0.2, score_type="pixel_max")
        self.assertEqual(record["prediction"], "uncalibrated")
        self.assertNotIn("threshold", record)
        self.assertEqual(record["detection_status"], "ok")
        self.assertEqual(record["num_defects"], 1)
        self.assertTrue(np.asarray(mask).any())
        _, _, manual = self.inspect(score_type="pixel_max", threshold=3.0)
        self.assertEqual(manual["score_type"], "pixel_max")
        self.assertEqual(manual["threshold"], 3.0)
        self.assertEqual(manual["prediction"], "normal")

    def test_manual_pixel_threshold_overrides_calibration_and_zero_is_valid(self):
        _, mask, record = self.inspect(pixel_threshold=3.0)
        self.assertEqual(record["pixel_threshold"], 3.0)
        self.assertEqual(record["detection_status"], "ok")
        self.assertEqual(record["detections"], [])
        self.assertFalse(np.asarray(mask).any())
        self.predictor.pixel_thresholds = {}
        _, _, zero = self.inspect(pixel_threshold=0.0)
        self.assertEqual(zero["pixel_threshold"], 0.0)
        self.assertGreater(zero["num_defects"], 0)

    def test_classification_threshold_can_be_manual_without_pixel_calibration(self):
        self.predictor.thresholds = {}
        self.predictor.pixel_thresholds = {}
        _, mask, record = self.inspect(threshold=0.8, score_type="image_score")
        self.assertEqual(record["prediction"], "defect")
        self.assertEqual(record["threshold"], 0.8)
        self.assertEqual(record["detection_status"], "uncalibrated")
        self.assertIsNone(mask)

    def test_cli_saves_readable_boxes_mask_raw_response_and_json(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "wafer.png"
            self.image.save(source)
            output = base / "predictions"
            arguments = ["custom_infer.py", "--image", str(source), "--class_name", "wafer",
                         "--save_path", str(base / "ckpt"), "--output", str(output),
                         "--min_area", "1", "--device", "cpu"]
            with patch("custom_infer.Predictor", return_value=self.predictor), \
                    patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                custom_infer.main()
            for suffix in ("_boxes.png", "_mask.png", "_overlay.png", "_heatmap.png"):
                with Image.open(output / (source.name + suffix)) as artifact:
                    self.assertEqual(artifact.size, self.image.size)
            np.testing.assert_array_equal(np.load(output / "wafer.png_raw.npy"), self.response)
            records = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["image"], str(source))
            self.assertEqual(records[0]["detections"][0]["bbox_xyxy"], [4, 2, 8, 6])
            self.predictor.predict.assert_called_once()

    def test_cli_uncalibrated_output_does_not_save_a_fabricated_zero_mask(self):
        self.predictor.pixel_thresholds = {}
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "wafer.png"
            self.image.save(source)
            output = base / "predictions"
            arguments = ["custom_infer.py", "--image", str(source), "--class_name", "wafer",
                         "--output", str(output)]
            with patch("custom_infer.Predictor", return_value=self.predictor), \
                    patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                custom_infer.main()
            self.assertTrue((output / "wafer.png_boxes.png").is_file())
            self.assertFalse((output / "wafer.png_mask.png").exists())
            record = json.loads((output / "predictions.json").read_text())[0]
            self.assertIsNone(record["detections"])
            self.assertEqual(record["detection_status"], "uncalibrated")

    def test_cli_uncalibrated_rerun_removes_previous_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "wafer.png"
            self.image.save(source)
            output = base / "predictions"
            arguments = ["custom_infer.py", "--image", str(source), "--class_name", "wafer",
                         "--output", str(output), "--min_area", "1"]
            with patch("custom_infer.Predictor", return_value=self.predictor), \
                    patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                custom_infer.main()
                self.assertTrue((output / "wafer.png_mask.png").exists())
                self.predictor.pixel_thresholds = {}
                custom_infer.main()
            self.assertFalse((output / "wafer.png_mask.png").exists())

    def test_cli_folder_preserves_subfolders_for_repeated_filenames(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "inputs"
            for name in ("batch1", "batch2"):
                (source / name).mkdir(parents=True)
                self.image.save(source / name / "wafer.png")
            output = base / "predictions"
            arguments = ["custom_infer.py", "--image", str(source), "--class_name", "wafer",
                         "--output", str(output), "--min_area", "1"]
            with patch("custom_infer.Predictor", return_value=self.predictor), \
                    patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                custom_infer.main()
            for name in ("batch1", "batch2"):
                self.assertTrue((output / name / "wafer.png_boxes.png").is_file())
                self.assertTrue((output / name / "wafer.png_mask.png").is_file())
            records = json.loads((output / "predictions.json").read_text())
            self.assertEqual(len(records), 2)

    def test_predictor_loads_matching_calibration_and_rejects_changed_inference_signature(self):
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        with tempfile.TemporaryDirectory() as folder, \
                patch("custom_infer.create_model", return_value=Mock()), \
                patch("custom_infer.AdaptedCLIP", return_value=model), \
                patch("custom_infer.torch.load", return_value={"image_adapter": {}, "text_adapter": {}}), \
                patch("custom_infer.setup_seed"), patch("custom_infer.torch.set_num_threads"):
            base = Path(folder)
            (base / "image_adapter.pth").write_bytes(b"image weights")
            (base / "text_adapter.pth").write_bytes(b"text weights")
            config = {"img_size": 56, "text_epoch": 1, "dataset": "MVTec"}
            (base / "config.json").write_text(json.dumps(config))
            initial = custom_infer.Predictor(base, device="cpu")
            calibration = {"inference_signature": initial.inference_signature(),
                           "thresholds": {"wafer": 0.5}, "score_type": "image_score",
                           "pixel_thresholds": {"wafer": 1.0}}
            (base / "thresholds.json").write_text(json.dumps(calibration))
            matching = custom_infer.Predictor(base, device="cpu")
            self.assertEqual(matching.thresholds, {"wafer": 0.5})
            self.assertEqual(matching.pixel_thresholds, {"wafer": 1.0})
            self.assertEqual(matching.score_type, "image_score")
            for option, value in (("dataset", "VisA"), ("image_adapt_weight", 0.2), ("quick_gelu", True)):
                with self.subTest(option=option):
                    (base / "config.json").write_text(json.dumps({**config, option: value}))
                    with self.assertWarnsRegex(UserWarning, "stale thresholds"):
                        changed = custom_infer.Predictor(base, device="cpu")
                    self.assertEqual(changed.thresholds, {})
                    self.assertEqual(changed.pixel_thresholds, {})
            (base / "config.json").write_text(json.dumps(config))
            with self.assertWarnsRegex(UserWarning, "stale thresholds"):
                resized = custom_infer.Predictor(base, img_size=70, device="cpu")
            self.assertEqual(resized.pixel_thresholds, {})
            (base / "text_adapter.pth").write_bytes(b"updated and different text weights")
            with self.assertWarnsRegex(UserWarning, "stale thresholds"):
                text_changed = custom_infer.Predictor(base, device="cpu")
            self.assertEqual(text_changed.thresholds, {})
            self.assertEqual(text_changed.pixel_thresholds, {})

    def test_web_handler_shares_detection_results_and_accepts_integer_valued_area(self):
        with patch("web_infer_app.gr.Interface", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)):
            interface = web_infer_app.build_demo(self.predictor)
        outputs = interface.fn(self.image, " wafer ", None, 1.0, True)
        boxes, mask, overlay, heatmap, record = outputs
        self.predictor.predict.assert_called_once_with(self.image, "wafer")
        self.assertEqual(record["detections"][0]["bbox_xyxy"], [4, 2, 8, 6])
        self.assertEqual(record["detection_settings"]["min_area"], 1)
        for artifact in (boxes, mask, overlay, heatmap):
            self.assertEqual(artifact.size, self.image.size)
        self.predictor.pixel_thresholds = {}
        _, no_mask, _, _, uncalibrated = interface.fn(self.image, "wafer", None, 1.0, True)
        self.assertIsNone(no_mask)
        self.assertIsNone(uncalibrated["detections"])
        self.assertEqual(uncalibrated["detection_status"], "uncalibrated")

    def test_web_omits_bottle_examples_when_optional_dataset_is_absent(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch("web_infer_app.ROOT", Path(folder)), \
                patch("web_infer_app.gr.Interface", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)):
            interface = web_infer_app.build_demo(self.predictor, class_name="bottle")
        self.assertEqual(interface.examples, [])

    def test_web_rejects_invalid_area_even_without_pixel_calibration(self):
        self.predictor.pixel_thresholds = {}
        with patch("web_infer_app.gr.Interface", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)):
            interface = web_infer_app.build_demo(self.predictor)
        for area in (None, 0, -1, 1.5, np.nan, np.inf, True):
            with self.subTest(area=area), self.assertRaises(web_infer_app.gr.Error):
                interface.fn(self.image, "wafer", None, area, True)


if __name__ == "__main__":
    unittest.main()
