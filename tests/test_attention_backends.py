import unittest

import torch

from model import EagerAttention


class TestAttentionBackends(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.eager = EagerAttention(
            hidden_size=32,
            num_heads=4,
            backend="eager",
        )
        self.sdpa = EagerAttention(
            hidden_size=32,
            num_heads=4,
            backend="sdpa",
        )
        self.sdpa.load_state_dict(self.eager.state_dict())

    def test_outputs_match(self):
        x = torch.randn(2, 8, 32)

        eager_output = self.eager(x)
        sdpa_output = self.sdpa(x)

        torch.testing.assert_close(
            sdpa_output,
            eager_output,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_input_gradients_match(self):
        x_eager = torch.randn(2, 8, 32, requires_grad=True)
        x_sdpa = x_eager.detach().clone().requires_grad_(True)

        self.sdpa.load_state_dict(self.eager.state_dict())

        self.eager(x_eager).square().mean().backward()
        self.sdpa(x_sdpa).square().mean().backward()

        torch.testing.assert_close(
            x_sdpa.grad,
            x_eager.grad,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_invalid_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported attention backend"):
            EagerAttention(
                hidden_size=32,
                num_heads=4,
                backend="invalid",
            )


if __name__ == "__main__":
    unittest.main()
