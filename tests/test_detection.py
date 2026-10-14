import unittest

import numpy as np
from PIL import Image

from defect_detection import calibrate_pixel_dice, calibrate_pixel_threshold, detect_defects, draw_detections


class DetectionTests(unittest.TestCase):
    def test_low_response_gradient_does_not_invent_boxes(self):
        response = np.linspace(0.1, 0.2, 120).reshape(10, 12)
        mask, regions = detect_defects(response, (120, 60), 0.5, min_area=1)
        self.assertEqual(regions, [])
        self.assertEqual(mask.shape, (60, 120))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertFalse(mask.any())

    def test_regions_have_exact_original_coordinates_scores_and_area(self):
        response = np.zeros((20, 40))
        response[3:7, 5:10] = 4.0
        response[14:17, 30:36] = 3.0
        response[0, 0] = 9.0  # Isolated noise is filtered.
        mask, regions = detect_defects(response, (40, 20), 2.0, min_area=3)
        self.assertEqual([r["bbox_xyxy"] for r in regions], [[5, 3, 10, 7], [30, 14, 36, 17]])
        self.assertEqual([r["area"] for r in regions], [20, 18])
        self.assertEqual(regions[0]["score_max"], 4.0)
        self.assertEqual(regions[0]["score_mean"], 4.0)
        self.assertAlmostEqual(regions[0]["area_fraction"], 20 / 800)
        self.assertEqual(np.count_nonzero(mask), 38)
        self.assertEqual(mask[0, 0], 0)

    def test_nonsquare_resize_maps_box_to_original_size(self):
        response = np.zeros((4, 4))
        response[1:3, 1:3] = 1.0
        mask, regions = detect_defects(response, (12, 8), 0.5, min_area=1)
        self.assertEqual(mask.shape, (8, 12))
        self.assertEqual(regions[0]["bbox_xyxy"], [3, 2, 9, 6])

    def test_constant_map_and_equality_use_raw_strict_threshold(self):
        response = np.full((4, 5), 7.0)
        _, equal = detect_defects(response, (5, 4), 7.0, min_area=1)
        mask, above = detect_defects(response, (5, 4), 6.0, min_area=1)
        self.assertEqual(equal, [])
        self.assertEqual(above[0]["bbox_xyxy"], [0, 0, 5, 4])
        self.assertEqual(np.count_nonzero(mask), 20)

    def test_morphology_and_fraction_filter(self):
        response = np.zeros((15, 20))
        response[5:10, 3:8] = 2.0
        response[5:10, 9:14] = 2.0
        response[1, 1] = 3.0
        _, separate = detect_defects(response, (20, 15), 1.0, min_area=2)
        mask, joined = detect_defects(response, (20, 15), 1.0, min_area=1,
                                     open_kernel=3, close_kernel=3, min_area_fraction=0.1)
        self.assertEqual(len(separate), 2)
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0]["bbox_xyxy"], [3, 5, 14, 10])
        self.assertEqual(joined[0]["area"], 55)
        self.assertEqual(mask[1, 1], 0)
        self.assertAlmostEqual(joined[0]["score_mean"], 100 / 55)

    def test_padding_clipping_and_max_regions_mask_consistency(self):
        response = np.zeros((6, 10))
        response[0:2, 0:3] = 5.0
        response[4:6, 8:10] = 2.0
        mask, regions = detect_defects(response, (10, 6), 1.0, min_area=1,
                                      padding=5, max_regions=1)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["bbox_xyxy"], [0, 0, 8, 6])
        self.assertEqual(np.count_nonzero(mask), 6)
        self.assertEqual(mask[4, 8], 0)

    def test_morphological_closing_preserves_defects_touching_image_boundary(self):
        response = np.zeros((10, 12))
        response[3:8, :4] = 2.0
        mask, regions = detect_defects(response, (12, 10), 1.0, min_area=1, close_kernel=3)
        self.assertEqual(regions[0]["bbox_xyxy"], [0, 3, 4, 8])
        self.assertEqual(regions[0]["area"], 20)
        self.assertTrue(np.all(mask[3:8, 0] == 255))

    def test_connectivity_controls_diagonal_components(self):
        response = np.eye(3)
        _, four = detect_defects(response, (3, 3), 0.5, min_area=1, connectivity=4)
        _, eight = detect_defects(response, (3, 3), 0.5, min_area=1, connectivity=8)
        self.assertEqual(len(four), 3)
        self.assertEqual(len(eight), 1)

    def test_drawing_preserves_original_image_and_exclusive_box(self):
        image = Image.new("RGB", (20, 10), (0, 0, 0))
        detections = [{"bbox_xyxy": [2, 3, 8, 7], "score_max": 4.0}]
        drawing = draw_detections(image, detections, show_scores=False, line_width=1)
        self.assertEqual(drawing.size, image.size)
        self.assertEqual(image.getpixel((2, 3)), (0, 0, 0))
        self.assertEqual(drawing.getpixel((2, 3)), (255, 40, 40))
        self.assertEqual(drawing.getpixel((7, 6)), (255, 40, 40))
        self.assertEqual(drawing.getpixel((8, 6)), (0, 0, 0))

    def test_calibration_budget_ties_and_zero_fpr(self):
        normal = [np.arange(10).reshape(2, 5)]
        calibration = calibrate_pixel_threshold(normal, 0.2)
        self.assertEqual(calibration["pixel_threshold"], 7.0)
        self.assertEqual(calibration["false_positive_pixels"], 2)
        self.assertEqual(calibration["observed_pixel_fpr"], 0.2)
        self.assertEqual(calibration["comparison"], ">")
        tied = calibrate_pixel_threshold([np.ones((2, 5))], 0.2)
        self.assertEqual(tied["false_positive_pixels"], 0)
        zero = calibrate_pixel_threshold(normal, 0)
        self.assertEqual(zero["pixel_threshold"], 9.0)
        self.assertEqual(zero["observed_pixel_fpr"], 0.0)

    def test_calibration_uses_inference_resize(self):
        response = np.arange(6).reshape(2, 3)
        calibration = calibrate_pixel_threshold([response], 0.1, image_sizes=[(9, 8)])
        mask, _ = detect_defects(response, (9, 8), calibration["pixel_threshold"], min_area=1)
        self.assertEqual(calibration["normal_pixels"], 72)
        self.assertEqual(calibration["resolution"], "original_image")
        self.assertEqual(np.count_nonzero(mask), calibration["false_positive_pixels"])
        self.assertLessEqual(np.count_nonzero(mask) / 72, 0.1)

    def test_supervised_dice_threshold_resizes_to_original_mask_dimensions(self):
        response = np.array([[0.0, 4.0], [0.0, 4.0]])
        target = np.zeros((4, 8), dtype=np.uint8)
        target[:, 4:] = 255
        calibration = calibrate_pixel_dice([response], [target], image_sizes=[(8, 4)])
        mask, regions = detect_defects(response, (8, 4), calibration["pixel_threshold"], min_area=1)
        self.assertEqual(calibration["validation_dice"], 1.0)
        self.assertEqual(calibration["num_pixels"], 32)
        self.assertEqual(calibration["num_foreground_pixels"], 16)
        self.assertEqual(calibration["observed_normal_pixel_fpr"], 0.0)
        self.assertEqual(regions[0]["bbox_xyxy"], [4, 0, 8, 4])
        np.testing.assert_array_equal(mask, target)
        inferred_size = calibrate_pixel_dice([response], [target])
        self.assertEqual(inferred_size["pixel_threshold"], calibration["pixel_threshold"])
        self.assertEqual(inferred_size["resolution"], "original_image")

    def test_supervised_dice_includes_background_from_normal_and_defective_images(self):
        response = [np.array([[0.0, 1.0, 2.0, 3.0]]), np.array([[4.0, 0.0, 0.0, 0.0]])]
        masks = [np.array([[0, 0, 1, 1]]), np.array([[0, 0, 0, 0]])]
        calibration = calibrate_pixel_dice(response, masks)
        self.assertEqual(calibration["pixel_threshold"], 1.0)
        self.assertEqual(calibration["validation_dice"], 0.8)
        self.assertEqual(calibration["validation_true_positive_pixels"], 2)
        self.assertEqual(calibration["validation_false_positive_pixels"], 1)
        self.assertEqual(calibration["num_background_pixels"], 6)
        self.assertAlmostEqual(calibration["observed_normal_pixel_fpr"], 1 / 6)

    def test_supervised_dice_cannot_split_equal_score_pixels(self):
        response = np.array([[0.0, 1.0, 1.0, 2.0]])
        target = np.array([[0, 1, 0, 1]])
        calibration = calibrate_pixel_dice([response], [target])
        self.assertEqual(calibration["pixel_threshold"], 0.0)
        self.assertEqual(calibration["comparison"], ">")
        self.assertEqual(calibration["validation_predicted_pixels"], 3)
        self.assertEqual(calibration["validation_dice"], 0.8)

    def test_supervised_dice_equal_optima_choose_higher_threshold(self):
        response = np.array([[0.0, 1.0, 1.0, 1.0, 2.0]])
        target = np.array([[0, 1, 0, 0, 1]])
        calibration = calibrate_pixel_dice([response], [target])
        self.assertEqual(calibration["pixel_threshold"], 1.0)
        self.assertAlmostEqual(calibration["validation_dice"], 2 / 3)
        self.assertEqual(calibration["validation_predicted_pixels"], 1)

    def test_supervised_dice_can_select_all_pixels_and_reports_missing_background(self):
        calibration = calibrate_pixel_dice([np.full((2, 3), 5.0)], [np.ones((2, 3), dtype=bool)])
        self.assertEqual(calibration["pixel_threshold"], np.nextafter(5.0, -np.inf))
        self.assertEqual(calibration["validation_dice"], 1.0)
        self.assertEqual(calibration["validation_predicted_pixels"], 6)
        self.assertIsNone(calibration["observed_normal_pixel_fpr"])

    def test_supervised_dice_matches_exhaustive_search_on_tied_random_scores(self):
        rng = np.random.default_rng(17)
        for sample in range(25):
            response = rng.integers(-2, 5, size=(5, 7)).astype(float)
            target = rng.integers(0, 2, size=(5, 7), dtype=np.uint8)
            target[0, 0] = 1
            foreground = target != 0
            candidates = np.append(np.unique(response), np.nextafter(response.min(), -np.inf))
            scores = []
            for threshold in candidates:
                predicted = response > threshold
                score = 2 * np.count_nonzero(predicted & foreground) / (predicted.sum() + foreground.sum())
                scores.append((float(score), float(threshold)))
            expected_dice, expected_threshold = max(scores)
            calibration = calibrate_pixel_dice([response], [target])
            with self.subTest(sample=sample):
                self.assertEqual(calibration["pixel_threshold"], expected_threshold)
                self.assertEqual(calibration["validation_dice"], expected_dice)

    def test_supervised_dice_rejects_missing_foreground_and_invalid_masks(self):
        response = np.zeros((2, 3))
        with self.assertRaisesRegex(ValueError, "foreground"):
            calibrate_pixel_dice([response], [np.zeros((2, 3))])
        for mask in (np.full((2, 3), 128), np.full((2, 3), 0.5),
                     np.zeros((2, 3, 3)), np.array([[np.nan]])):
            with self.subTest(mask=repr(mask)), self.assertRaises(ValueError):
                calibrate_pixel_dice([response], [mask])
        for maps, masks in (([], []), ([response], []), ([], [np.ones((2, 3))])):
            with self.assertRaises(ValueError):
                calibrate_pixel_dice(maps, masks)
        for sizes in ([], [(3, 2), (3, 2)], [(2, 3)]):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                calibrate_pixel_dice([response], [np.ones((2, 3))], image_sizes=sizes)

    def test_invalid_response_parameters_and_sizes_raise(self):
        for response in (np.zeros((0, 4)), np.zeros((1, 2, 3)), [[np.nan]], [[np.inf]], [["x"]]):
            with self.subTest(response=repr(response)), self.assertRaises(ValueError):
                detect_defects(response, (5, 4), 1.0)
        for kwargs in ({"pixel_threshold": np.inf}, {"open_kernel": 2}, {"close_kernel": -1},
                       {"min_area": 0}, {"min_area_fraction": 1.1}, {"padding": -1},
                       {"connectivity": 6}, {"max_regions": 0}):
            options = {"pixel_threshold": 1.0, **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                detect_defects(np.zeros((4, 5)), (5, 4), **options)
        for sizes in ([], [(5, 4), (5, 4)]):
            with self.assertRaises(ValueError):
                calibrate_pixel_threshold([np.zeros((4, 5))], image_sizes=sizes)
        with self.assertRaises(ValueError):
            calibrate_pixel_threshold([])
        with self.assertRaises(ValueError):
            calibrate_pixel_threshold([np.zeros((4, 5))], 1.0)


if __name__ == "__main__":
    unittest.main()
