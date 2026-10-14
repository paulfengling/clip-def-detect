import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from data_preparation import validate_record
from dataset import BaseDataset, BaseSingleClassDataset, get_dataset


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGB", (28, 14), (20, 40, 60)).save(self.root / "normal.png")
        Image.new("RGB", (28, 14), (70, 90, 110)).save(self.root / "defect.png")
        mask = np.zeros((14, 28), dtype=np.uint8)
        mask[3:8, 7:12] = 255
        Image.fromarray(mask).save(self.root / "mask.png")
        self.rows = [
            {"class_name": "wafer", "image_path": "normal.png", "label": 0},
            {"class_name": "wafer", "image_path": "defect.png", "label": 1, "mask_path": "mask.png"},
        ]
        self.meta = self.root / "train.jsonl"

    def write_rows(self, blanks=False):
        separator = "\n\n" if blanks else "\n"
        self.meta.write_text(separator.join(json.dumps(row) for row in self.rows) + "\n",
                             encoding="utf-8-sig")

    def load(self, stage="train"):
        return get_dataset("CustomValidation", 56, "few_shot", stage=stage,
                           data_path=str(self.root), meta_path=str(self.meta))

    def test_blank_lines_bom_and_missing_normal_mask_load_for_train_and_eval(self):
        self.write_rows(blanks=True)
        with patch("dataset.validate_record", wraps=validate_record) as validate:
            text, image = self.load()
            self.assertEqual(validate.call_count, 2)
        self.assertEqual(len(text), 2)
        self.assertEqual(len(image), 2)
        image.transforms_list = []
        self.assertEqual(tuple(image[0]["mask"].shape), (1, 56, 56))
        self.assertEqual(image[0]["mask"].sum().item(), 0)
        self.assertGreater(image[1]["mask"].sum().item(), 0)
        self.assertEqual(len(self.load("test")["wafer"]), 2)
        self.assertEqual(len(BaseDataset(self.root, self.meta, 56)), 2)
        self.assertEqual(len(BaseSingleClassDataset(self.root, self.meta, 56, "wafer")), 2)

    def test_invalid_product_names_and_labels_are_rejected_before_iteration(self):
        original = self.rows[0].copy()
        for key, value in [("class_name", None), ("class_name", " "),
                           ("class_name", 7), ("label", 2), ("label", "1")]:
            self.rows[0] = {**original, key: value}
            self.write_rows()
            with self.subTest(key=key, value=value):
                with self.assertRaisesRegex(ValueError, "Invalid metadata.*:1"):
                    self.load()
        for key in ("class_name", "image_path", "label"):
            self.rows[0] = original.copy()
            del self.rows[0][key]
            self.write_rows()
            with self.subTest(missing_key=key):
                with self.assertRaisesRegex(ValueError, "Invalid metadata"):
                    self.load("test")

    def test_empty_or_malformed_manifests_are_rejected(self):
        for content, message in [("\n \n", "Empty dataset"),
                                 ("[]\n", "Each JSONL row must be an object"),
                                 ("{bad json}\n", "Invalid metadata")]:
            self.meta.write_text(content, encoding="utf-8")
            with self.subTest(content=content):
                with self.assertRaisesRegex(ValueError, message):
                    self.load()

    def test_missing_images_masks_empty_masks_and_bad_masks_fail_at_startup(self):
        for mask, message in [
            (Image.new("L", (28, 14), 0), "Empty defect mask"),
            (Image.new("L", (14, 14), 255), "size mismatch"),
            (Image.new("RGB", (28, 14), "white"), "grayscale"),
            (Image.new("L", (28, 14), 120), "non-binary"),
        ]:
            mask.save(self.root / "mask.png")
            self.write_rows()
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    self.load()
        del self.rows[1]["mask_path"]
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "no mask"):
            self.load()
        self.rows[0]["image_path"] = "missing.png"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "Invalid metadata.*missing.png"):
            self.load()

    def test_zero_one_mask_is_supported_and_normal_mask_is_rejected(self):
        mask = np.zeros((14, 28), dtype=np.uint8)
        mask[2:6, 3:9] = 1
        Image.fromarray(mask).save(self.root / "mask.png")
        self.write_rows()
        self.assertEqual(len(self.load()[0]), 2)
        self.rows[0]["mask_path"] = "mask.png"
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "Normal images should have an empty mask_path"):
            self.load()


if __name__ == "__main__":
    unittest.main()
