import io
import unittest

import numpy as np
import torch

from torch_fiber import DFCAdamW, TorchSignFiberChannel
from vision_benchmark import CIFAR4BitCodec


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

    def test_payload_can_be_rewritten_between_optimizer_steps(self):
        torch.manual_seed(41)
        reference = torch.nn.Linear(31, 17)
        lifted = torch.nn.Linear(31, 17)
        lifted.load_state_dict(reference.state_dict())
        opt_ref = DFCAdamW(reference.parameters(), lr=7e-4, enable_fiber=False)
        opt_dfc = DFCAdamW(lifted.parameters(), lr=7e-4, enable_fiber=True)
        channel = TorchSignFiberChannel(opt_dfc)
        rng = np.random.default_rng(43)
        for _ in range(12):
            start = int(rng.integers(0, max(1, channel.byte_capacity - 19)))
            payload = rng.bytes(min(19, channel.byte_capacity - start))
            channel.write_bytes(start, payload)
            x, y = torch.randn(13, 31), torch.randn(13, 17)
            for model, optimizer in ((reference, opt_ref), (lifted, opt_dfc)):
                optimizer.zero_grad()
                torch.nn.functional.mse_loss(model(x), y).backward()
                optimizer.step()
            self.assertEqual(channel.read_bytes(start, len(payload)), payload)
        for a, b in zip(reference.parameters(), lifted.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_cifar_codec_shape_crc_and_error_bound(self):
        torch.manual_seed(47)
        x = torch.rand(3, 32, 32)
        teacher = torch.linspace(-2, 2, 100)
        raw = CIFAR4BitCodec.encode(x, 73, 7, teacher)
        restored, label, task, logits = CIFAR4BitCodec.decode(raw)
        self.assertEqual(len(raw), CIFAR4BitCodec.RECORD_BYTES)
        self.assertEqual(tuple(restored.shape), (3, 32, 32))
        self.assertEqual((label, task), (73, 7))
        self.assertTrue(torch.all((restored >= 0) & (restored <= 1)))
        self.assertTrue(torch.equal(logits, teacher.to(torch.float16).float()))
        corrupted = bytearray(raw)
        corrupted[101] ^= 1
        with self.assertRaises(RuntimeError):
            CIFAR4BitCodec.decode(bytes(corrupted))


if __name__ == "__main__":
    unittest.main()
