import unittest

import torch

from src.train import build_model


class TestHybridSolarModel(unittest.TestCase):
    def test_build_model_supports_cnn_wnn_mmha(self):
        cfg = {
            "model": {
                "type": "cnn_wnn_mmha",
                "hidden_size": 32,
                "num_layers": 2,
                "dropout": 0.1,
                "num_heads": 4,
            }
        }

        model = build_model(cfg, input_size=16, sequence_length=24)
        x = torch.randn(4, 24, 16)
        y = model(x)

        self.assertEqual(y.shape, (4,))


if __name__ == "__main__":
    unittest.main()
