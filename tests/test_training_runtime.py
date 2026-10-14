"""Exercise adapter gradients, surgery attention and available runtime devices."""
import copy
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from forward_utils import calculate_similarity_map
from model.adapter import AdaptedCLIP
from model.clip import create_model
from model.model import CLIP
from model.runtime import resolve_device
from model.transformer import Attention, QuickGELU, Transformer
from train import train_image_adapter, train_text_adapter
from utils import setup_seed


def tiny_clip():
    return CLIP(
        embed_dim=16,
        vision_cfg={"image_size": 56, "patch_size": 14, "width": 32,
                    "head_width": 8, "layers": 4, "patch_dropout": 0},
        text_cfg={"context_length": 8, "vocab_size": 32, "width": 32,
                  "heads": 4, "layers": 3},
        quick_gelu=True,
    ).eval()


def tiny_text_embedding(model, dataset_name, class_name, device):
    # The two rows are normal and defective prompts with a highest-ID EOT.
    tokens = torch.tensor([[1, 2, 31, 0, 0, 0, 0, 0],
                           [1, 3, 4, 31, 0, 0, 0, 0]], device=device)
    return F.normalize(model.encode_text(tokens), dim=-1).T


class TrainingRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.deterministic_before = torch.are_deterministic_algorithms_enabled()
        self.warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()

    def tearDown(self):
        torch.use_deterministic_algorithms(self.deterministic_before,
                                          warn_only=self.warn_only_before)

    def test_seed_uses_strict_cpu_and_explicit_mps_warning_mode(self):
        setup_seed(123, device="cpu")
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        with self.assertLogs("utils", level="WARNING") as logs:
            setup_seed(123, device="mps")
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertTrue(torch.is_deterministic_algorithms_warn_only_enabled())
        self.assertIn("bitwise reproducibility is not guaranteed", logs.output[0])
        setup_seed(123, device="cpu")
        self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())

    def test_device_auto_priority_and_explicit_unavailable_devices(self):
        with patch("torch.cuda.is_available", return_value=True), patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(str(resolve_device()), "cuda:0")
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(str(resolve_device()), "mps")
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(str(resolve_device()), "cpu")
            for device in ("mps", "cuda"):
                with self.assertRaises(ValueError):
                    resolve_device(device)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_seeded_mps_text_eot_gather_backward(self):
        setup_seed(123, device="mps")
        model = AdaptedCLIP(tiny_clip(), text_adapt_until=2,
                            image_adapt_until=2).to("mps").eval()
        tokens = torch.tensor([[1, 2, 31, 0, 0, 0, 0, 0],
                               [1, 3, 31, 0, 0, 0, 0, 0]], device="mps")
        model.encode_text(tokens).square().mean().backward()
        gradient = model.text_adapter[0].fc[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0)

    def test_surgery_attention_is_independent_between_images(self):
        torch.manual_seed(10)
        attention = Attention(16, 16, num_heads=4, qkv_bias=True).eval()
        tokens = torch.randn(7, 2, 16)
        batched, weights = attention(tokens, tokens, tokens)
        separately = torch.cat([attention(tokens[:, i:i + 1], None, None)[0]
                                for i in range(2)], dim=1)
        torch.testing.assert_close(batched, separately)
        self.assertEqual(weights.shape, (2, 4, 7, 7))
        changed = tokens.clone()
        changed[1, 0] += 10
        result, _ = attention(changed, None, None)
        torch.testing.assert_close(result[:, 1], batched[:, 1])
        self.assertFalse(torch.allclose(result[0, 0], batched[0, 0]))

    def test_transformer_checkpointing_preserves_features_and_gradients(self):
        torch.manual_seed(11)
        regular = Transformer(16, layers=3, heads=4).eval()
        checkpointed = copy.deepcopy(regular)
        checkpointed.grad_checkpointing = True
        input_regular = torch.randn(5, 2, 16, requires_grad=True)
        input_checkpointed = input_regular.detach().clone().requires_grad_()
        expected, features = regular(input_regular, [1, 3])
        actual, checkpoint_features = checkpointed(input_checkpointed, [1, 3])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(len(checkpoint_features), 2)
        for left, right in zip(features, checkpoint_features):
            torch.testing.assert_close(left, right)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(input_regular.grad, input_checkpointed.grad)

    def test_local_state_dict_load_preserves_weights_and_activation_setting(self):
        config = {"embed_dim": 16,
                  "vision_cfg": {"image_size": 56, "patch_size": 14, "width": 32,
                                 "head_width": 8, "layers": 4, "patch_dropout": 0},
                  "text_cfg": {"context_length": 8, "vocab_size": 32, "width": 32,
                               "heads": 4, "layers": 3}}
        original = tiny_clip()
        with tempfile.TemporaryDirectory() as folder:
            filename = Path(folder) / "clip.pth"
            torch.save({"state_dict": original.state_dict()}, filename)
            with patch.dict("model.clip._MODEL_CONFIGS", {"Tiny-Test": config}), \
                 patch.dict("model.clip._MODEL_CKPT_PATHS", {"Tiny-Test": filename}):
                for quick_gelu in (False, True):
                    loaded = create_model("Tiny-Test", img_size=56, pretrained="openai",
                                          require_pretrained=True, force_quick_gelu=quick_gelu)
                    activation = QuickGELU if quick_gelu else torch.nn.GELU
                    self.assertIsInstance(loaded.transformer.resblocks[0].mlp.gelu, activation)
                    torch.testing.assert_close(loaded.visual.conv1.weight, original.visual.conv1.weight)

    def test_two_stages_update_adapters_with_frozen_backbone(self):
        devices = [torch.device("cpu")]
        if torch.backends.mps.is_available():
            devices.append(torch.device("mps"))
        if torch.cuda.is_available():
            devices.append(torch.device("cuda:0"))
        for device in devices:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as folder:
                setup_seed(12, device=device)
                backbone = tiny_clip().to(device)
                surgery = copy.deepcopy(backbone)
                surgery.visual.DAPM_replace(4)
                surgery.eval().requires_grad_(False)
                self.assertTrue(all(p.device.type == device.type for p in surgery.parameters()))
                model = AdaptedCLIP(backbone, text_adapt_until=2,
                                    image_adapt_until=2, relu=False).to(device).eval()
                model.grad_checkpointing = True
                backbone_before = model.clipmodel.visual.conv1.weight.detach().clone()
                text_before = model.text_adapter[-1].fc[0].weight.detach().clone()
                image_before = model.image_adapter["seg_proj"][0].fc.weight.detach().clone()
                samples = []
                for label in (0, 1):
                    mask = torch.zeros(1, 56, 56)
                    if label:
                        mask[:, 14:28, 14:28] = 1
                    samples.append({"image": torch.randn(3, 56, 56), "mask": mask,
                                    "class_name": "wafer", "label": label})
                loader = DataLoader(samples, batch_size=2)
                logger = logging.getLogger("runtime_test")
                logger.addHandler(logging.NullHandler())
                optimizer = torch.optim.Adam(model.text_adapter.parameters(), lr=1e-3)
                with patch("train.get_adapted_single_class_text_embedding", side_effect=tiny_text_embedding):
                    train_text_adapter(model, surgery, 0.1, loader, optimizer, device,
                                       0, folder, 1, "Custom", 56, logger)
                self.assertFalse(torch.equal(text_before, model.text_adapter[-1].fc[0].weight))
                with torch.no_grad():
                    text = {"wafer": tiny_text_embedding(model, "Custom", "wafer", device)}
                optimizer = torch.optim.Adam(model.image_adapter.parameters(), lr=1e-3)
                scheduler = MultiStepLR(optimizer, milestones=[10])
                train_image_adapter(model, text, loader, optimizer, scheduler, device,
                                    0, folder, 1, 56, logger)
                self.assertFalse(torch.equal(image_before, model.image_adapter["seg_proj"][0].fc.weight))
                torch.testing.assert_close(backbone_before, model.clipmodel.visual.conv1.weight)
                self.assertTrue(all(p.grad is None for p in model.clipmodel.parameters()))
                for name, key in (("text_adapter.pth", "text_optimizer"),
                                  ("image_adapter.pth", "image_optimizer")):
                    saved = torch.load(Path(folder) / name, map_location="cpu", weights_only=True)
                    self.assertEqual(saved["epoch"], 1)
                    self.assertIn(key, saved)
                with torch.inference_mode():
                    features, detection = model(torch.randn(1, 3, 56, 56, device=device))
                    score_map = calculate_similarity_map(features[0], text["wafer"], 56, test=True, domain="Industrial")
                    self.assertEqual(score_map.shape, (1, 1, 56, 56))
                    self.assertTrue(torch.isfinite(score_map).all())
                    self.assertTrue(torch.isfinite(detection).all())


if __name__ == "__main__":
    unittest.main()
