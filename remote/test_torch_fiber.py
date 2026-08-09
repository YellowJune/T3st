import io
import unittest

import numpy as np
import torch

from torch_fiber import DFCAdamW, TorchSignFiberChannel


class TorchFiberTests(unittest.TestCase):
    def test_reference_and_payload_trajectory_are_bitwise_equal(self):
        torch.manual_seed(17)
        reference = torch.nn.Linear(257, 31)
        lifted = torch.nn.Linear(257, 31)
        lifted.load_state_dict(reference.state_dict())
        opt_ref = DFCAdamW(reference.parameters(), lr=3e-4, weight_decay=0.01, enable_fiber=False)
        opt_dfc = DFCAdamW(lifted.parameters(), lr=3e-4, weight_decay=0.01, enable_fiber=True)
        channel = TorchSignFiberChannel(opt_dfc)
        rng = np.random.default_rng(19)
        payload = rng.bytes(channel.byte_capacity)
        channel.write_bytes(0, payload)
        for _ in range(50):
            x = torch.randn(23, 257)
            y = torch.randn(23, 31)
            for model, optimizer in ((reference, opt_ref), (lifted, opt_dfc)):
                optimizer.zero_grad()
                torch.nn.functional.mse_loss(model(x), y).backward()
                optimizer.step()
        for a, b in zip(reference.parameters(), lifted.parameters()):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(channel.read_bytes(0, len(payload)), payload)

    def test_checkpoint_preserves_payload(self):
        model = torch.nn.Linear(128, 64)
        optimizer = DFCAdamW(model.parameters(), enable_fiber=True)
        channel = TorchSignFiberChannel(optimizer)
        payload = bytes((index * 37) % 256 for index in range(channel.byte_capacity))
        channel.write_bytes(0, payload)
        file = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, file)
        file.seek(0)
        restored_model = torch.nn.Linear(128, 64)
        restored_optimizer = DFCAdamW(restored_model.parameters(), enable_fiber=True)
        checkpoint = torch.load(file, weights_only=True)
        restored_model.load_state_dict(checkpoint["model"])
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        restored = TorchSignFiberChannel(restored_optimizer)
        self.assertEqual(restored.read_bytes(0, len(payload)), payload)


if __name__ == "__main__":
    unittest.main()
