import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from custom_infer import find_image_checkpoint, render_prediction
from forward_utils import metrics_eval
from prepare_fewshot import make_split

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_numeric_checkpoint_order_and_latest_alias(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ("image_adapter_9.pth", "image_adapter_20.pth"):
                (Path(folder) / name).touch()
            self.assertEqual(find_image_checkpoint(folder).name, "image_adapter_20.pth")
            (Path(folder) / "image_adapter.pth").touch()
            self.assertEqual(find_image_checkpoint(folder).name, "image_adapter.pth")

    def test_constant_heatmap_preserves_nonsquare_image_dimensions(self):
        image = Image.new("RGB", (160, 90))
        overlay, heatmap = render_prediction(image, np.full((56, 56), 3.0, dtype=np.float32))
        self.assertEqual(overlay.size, image.size)
        self.assertEqual(heatmap.size, image.size)
        self.assertTrue(np.isfinite(np.asarray(overlay)).all())

    def test_fewshot_split_has_no_leakage_with_generated_mvtec_layout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split, kind, count in (("train", "good", 8), ("test", "good", 10),
                                       ("test", "crack", 7), ("test", "stain", 7)):
                images = root / "wafer" / split / kind
                images.mkdir(parents=True)
                for index in range(count):
                    Image.new("RGB", (8, 8), (index, 20, 30)).save(images / f"{index}.png")
                    if kind != "good":
                        masks = root / "wafer" / "ground_truth" / kind
                        masks.mkdir(parents=True, exist_ok=True)
                        Image.new("L", (8, 8), 255).save(masks / f"{index}_mask.png")
            split = make_split(root, ["wafer"])
            self.assertEqual([len(split[key]) for key in ("train", "val", "test")], [8, 8, 12])
            self.assertEqual(sum(row["label"] for row in split["train"]), 4)
            paths = [{row["image_path"] for row in rows} for rows in split.values()]
            for index, selected in enumerate(paths):
                for earlier in paths[:index]:
                    self.assertFalse(selected & earlier)
            self.assertEqual(split, make_split(root, ["wafer"]))

    @unittest.skipUnless((ROOT / "data/opencvxuetang/MVTecAD/bottle").is_dir(),
                         "Optional local MVTec bottle dataset is not included; add it to run this integration test")
    def test_local_bottle_fewshot_split_has_no_leakage(self):
        split = make_split(ROOT / "data/opencvxuetang/MVTecAD", ["bottle"])
        self.assertEqual(len(split["train"]), 8)
        self.assertEqual(sum(r["label"] for r in split["train"]), 4)
        paths = [{r["image_path"] for r in rows} for rows in split.values()]
        for i in range(len(paths)):
            for j in range(i):
                self.assertFalse(paths[i] & paths[j])
        self.assertEqual(split, make_split(ROOT / "data/opencvxuetang/MVTecAD", ["bottle"]))

    def test_degenerate_metrics_do_not_invent_accuracy(self):
        result = metrics_eval(np.zeros((2, 1, 4, 4)), np.zeros(2),
                              np.full((2, 4, 4), 2.0), np.full(2, 0.5), "bottle", "Industrial")
        self.assertTrue(np.isnan(result["pixel AUC"]))
        self.assertTrue(np.isnan(result["image AUC"]))


if __name__ == "__main__":
    unittest.main()
