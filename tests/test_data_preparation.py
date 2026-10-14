import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

import numpy as np
from PIL import Image

from create_masks import convert_annotation
from data_preparation import audit_splits, validate_record
from split_dataset import collect_records, split_records
from export_sam3_mask import save_sam3_mask


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGB", (12, 8), (30, 50, 70)).save(self.root / "sample.png")
        self.data = {"imagePath": "sample.png", "imageWidth": 12, "imageHeight": 8, "shapes": [
            {"label": "defect", "shape_type": "polygon", "points": [[1, 1], [4, 1], [4, 4], [1, 4]]},
            {"label": "defect", "shape_type": "rectangle", "points": [[10, 6], [8, 5]]},
        ]}

    def convert(self):
        annotation = self.root / "sample.json"
        annotation.write_text(json.dumps(self.data), encoding="utf-8")
        return convert_annotation(annotation, self.root, self.root, {"defect"})

    def test_multiple_regions_preserve_dimensions_and_binary_values(self):
        _, mask, overlay = self.convert()
        self.assertEqual(mask.size, (12, 8))
        self.assertEqual(mask.mode, "L")
        self.assertEqual(overlay.size, mask.size)
        values = np.asarray(mask)
        self.assertEqual(set(np.unique(values)), {0, 255})
        self.assertEqual(values[2, 2], 255)
        self.assertEqual(values[5, 9], 255)
        self.assertEqual(values[0, 0], 0)

    def test_unknown_labels_and_empty_annotations_are_rejected(self):
        self.data["shapes"][0]["label"] = "whole_object"
        with self.assertRaisesRegex(ValueError, "Unexpected label"):
            self.convert()
        self.data["shapes"] = []
        with self.assertRaisesRegex(ValueError, "No annotated defect"):
            self.convert()

    def test_annotation_size_mismatch_and_outside_points_are_rejected(self):
        self.data["imageWidth"] = 99
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            self.convert()
        self.data["imageWidth"] = 12
        self.data["shapes"][0]["points"][0] = [-3, 0]
        with self.assertRaisesRegex(ValueError, "outside image"):
            self.convert()

    def test_mask_validation_rejects_empty_color_and_wrong_size(self):
        row = {"image_path": "sample.png", "mask_path": "mask.png", "label": 1}
        for image, message in [(Image.new("L", (12, 8)), "Empty defect mask"),
                               (Image.new("L", (8, 8), 255), "size mismatch"),
                               (Image.new("RGB", (12, 8), "white"), "grayscale"),
                               (Image.new("L", (12, 8), 100), "non-binary")]:
            image.save(self.root / "mask.png")
            with self.assertRaisesRegex(ValueError, message):
                validate_record(self.root, row)

    def test_duplicate_content_is_caught_even_with_different_names(self):
        with Image.open(self.root / "sample.png") as source:
            source.save(self.root / "duplicate.png")
        splits = {"train": [{"class_name": "bottle", "image_path": "sample.png", "label": 0}],
                  "test": [{"class_name": "bottle", "image_path": "duplicate.png", "label": 0}]}
        with self.assertRaisesRegex(ValueError, "Duplicate image content"):
            audit_splits(self.root, splits)

    def test_sam3_requires_selection_and_exports_selected_union(self):
        candidates = np.zeros((3, 1, 8, 12), dtype=bool)
        candidates[0, 0, 1:3, 1:3] = True
        candidates[1, 0, 4:6, 6:8] = True
        candidates[2, 0, 0, 11] = True
        output = self.root / "sam_mask.png"
        with self.assertRaisesRegex(ValueError, "explicitly select"):
            save_sam3_mask(candidates, self.root / "sample.png", output)
        mask_path, preview = save_sam3_mask(candidates, self.root / "sample.png", output, indices=[0, 1])
        with Image.open(mask_path) as mask:
            values = np.asarray(mask)
            self.assertEqual(mask.size, (12, 8))
            self.assertEqual(values[1, 1], 255)
            self.assertEqual(values[4, 6], 255)
            self.assertEqual(values[0, 11], 0)
        self.assertTrue(preview.is_file())
        with self.assertRaises(FileExistsError):
            save_sam3_mask(candidates, self.root / "sample.png", output, indices=[0])

    def test_sam3_rejects_empty_probabilities_and_wrong_dimensions(self):
        for masks, message in [(np.zeros((0, 1, 8, 12)), "nonempty"),
                               (np.full((8, 12), 0.6), "binary"),
                               (np.ones((4, 6)), "ORIGINAL"),
                               (np.zeros((8, 12)), "no defect pixels")]:
            with self.assertRaisesRegex(ValueError, message):
                save_sam3_mask(masks, self.root / "sample.png", self.root / "mask.png")


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"image_path": f"part/{label}/{i % 3}/{i}.png", "class_name": "part", "label": label, "mask_path": ""}
                     for label in (0, 1) for i in range(20)]

    def test_fewshot_counts_reproducibility_and_disjointness(self):
        split = split_records(self.rows, shot=8, val_per_label=4)
        self.assertEqual([len(split[key]) for key in ("train", "val", "test")], [8, 8, 24])
        self.assertEqual(sum(row["label"] for row in split["train"]), 4)
        self.assertEqual(split, split_records(self.rows, shot=8, val_per_label=4))
        paths = [row["image_path"] for rows in split.values() for row in rows]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(len(paths), len(self.rows))

    def test_ratios_and_too_few_samples(self):
        split = split_records(self.rows, ratios=[0.7, 0.15, 0.15])
        self.assertEqual([len(split[key]) for key in ("train", "val", "test")], [28, 6, 6])
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            split_records(self.rows, ratios=[0.7, 0.2, 0.2])
        with self.assertRaisesRegex(ValueError, "Not enough"):
            split_records(self.rows, shot=40)

    def test_custom_layout_is_train_loader_compatible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for path in ("part/good/a.png", "part/defect/crack/b.png", "part/masks/crack/b_mask.png"):
                destination = root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (8, 8), 255 if "mask" in path else 0).save(destination)
            rows = collect_records(root, ["part"])
            self.assertEqual(rows[0]["mask_path"], "")
            self.assertEqual(rows[1]["mask_path"], "part/masks/crack/b_mask.png")
            self.assertEqual(validate_record(root, rows[1])["defect_fraction"], 1.0)

    def test_annotation_to_training_manifest_cli(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            good = root / "part/good"
            defect = root / "part/defect/crack"
            good.mkdir(parents=True)
            defect.mkdir(parents=True)
            for i in range(12):
                Image.new("RGB", (16, 10), (i, 20, 30)).save(good / f"{i}.png")
                Image.new("RGB", (16, 10), (i, 60, 90)).save(defect / f"{i}.png")
                annotation = {"imagePath": f"{i}.png", "imageWidth": 16, "imageHeight": 10,
                              "shapes": [{"label": "defect", "shape_type": "polygon", "points": [[1, 1], [4, 1], [4, 4]]}]}
                (defect / f"{i}.json").write_text(json.dumps(annotation), encoding="utf-8")
            subprocess.run([sys.executable, str(project / "create_masks.py"), "--annotations", str(root / "part/defect"),
                            "--output", str(root / "part/masks")], check=True, capture_output=True)
            subprocess.run([sys.executable, str(project / "split_dataset.py"), "--data_root", str(root), "--classes", "part",
                            "--shot", "8", "--output", str(root / "metadata")], check=True, capture_output=True)
            for split in ("train", "val", "test"):
                rows = [json.loads(line) for line in (root / "metadata" / f"{split}.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 8)
                for row in rows:
                    validate_record(root, row)
            self.assertEqual(len(json.loads((root / "metadata/audit.json").read_text())), 24)


if __name__ == "__main__":
    unittest.main()
