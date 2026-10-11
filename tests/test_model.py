import unittest
from pathlib import Path

import torch
import yaml

from model import TransformerLM


ROOT = Path(__file__).resolve().parents[1]


def small_model():
    """Small configuration for fast CPU correctness tests."""
    return TransformerLM(
        vocab_size=128,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=64,
        max_seq_len=32,
    )


class TestTransformerLM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = small_model()

    def test_forward_output_shape(self):
        input_ids = torch.randint(0, 128, (2, 8))

        logits, loss = self.model(input_ids)

        self.assertEqual(tuple(logits.shape), (2, 8, 128))
        self.assertIsNone(loss)
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_loss_is_finite(self):
        input_ids = torch.randint(0, 128, (2, 8))
        labels = torch.randint(0, 128, (2, 8))

        _, loss = self.model(input_ids, labels)

        self.assertIsNotNone(loss)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_gradients_are_finite(self):
        input_ids = torch.randint(0, 128, (2, 8))
        labels = torch.randint(0, 128, (2, 8))

        _, loss = self.model(input_ids, labels)
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        self.assertTrue(gradients)
        self.assertTrue(all(grad is not None for grad in gradients))
        self.assertTrue(
            all(torch.isfinite(grad).all().item() for grad in gradients)
        )

    def test_optimizer_updates_parameters(self):
        input_ids = torch.randint(0, 128, (2, 8))
        labels = torch.randint(0, 128, (2, 8))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        parameter = self.model.blocks[0].attention.qkv.weight
        before = parameter.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        _, loss = self.model(input_ids, labels)
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.equal(before, parameter.detach()))

    def test_sequence_limit_is_enforced(self):
        input_ids = torch.randint(0, 128, (1, 33))

        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            self.model(input_ids)

    def test_token_embedding_and_lm_head_weights_are_tied(self):
        self.assertIs(
            self.model.token_embedding.weight,
            self.model.lm_head.weight,
        )


class TestBaselineConfig(unittest.TestCase):
    def test_baseline_config_loads(self):
        config_path = ROOT / "configs" / "baseline.yaml"

        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)

        self.assertIn("model", config)
        self.assertIn("training", config)
        self.assertEqual(config["model"]["hidden_size"], 768)
        self.assertEqual(config["model"]["num_layers"], 6)
        self.assertEqual(config["attention"]["backend"], "eager")


if __name__ == "__main__":
    unittest.main()
