from __future__ import annotations

import unittest
from pathlib import Path

import torch

try:
    from gpudrive_cuda_torch import TorchDriveSim
except ImportError:
    TorchDriveSim = None


@unittest.skipUnless(
    TorchDriveSim is not None and torch.cuda.is_available(),
    "requires the compiled Torch CUDA bridge",
)
class TorchBridgeIntegrationTest(unittest.TestCase):
    def test_shapes_stream_and_selective_reset(self):
        root = Path(__file__).resolve().parents[2]
        runtime = root / "dataset/nuplan/rl_runtime/mock_nuplan_00100000"
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            simulator = TorchDriveSim([str(runtime)], 2, 0)
            frame = simulator.reset()
            self.assertEqual(tuple(frame["states"].shape), (2, 64, 5))
            self.assertEqual(tuple(frame["partners"].shape), (2, 64, 16, 9))
            self.assertEqual(tuple(frame["map"].shape), (2, 64, 64, 7))
            self.assertEqual(tuple(frame["events"].shape), (2, 64, 5))
            self.assertTrue(all(value.is_cuda for value in frame.values()))

            modes = torch.zeros(2, 64, dtype=torch.uint8, device="cuda:0")
            modes[:, 1] = 1
            simulator.set_control_modes(modes)
            actions = torch.zeros(2, 64, 2, device="cuda:0")
            stepped = simulator.step(actions)
            reset_mask = torch.tensor([1, 0], dtype=torch.uint8, device="cuda:0")
            reset = simulator.reset_worlds(reset_mask)
            marker = reset["states"].sum() + stepped["states"].sum()
        stream.synchronize()

        self.assertTrue(torch.isfinite(marker))
        self.assertEqual(int(reset["world_step"][0]), 0)
        self.assertEqual(int(reset["world_step"][1]), 1)
        self.assertEqual(int(reset["control_modes"][0, 1]), 0)
        self.assertEqual(int(reset["control_modes"][1, 1]), 1)


if __name__ == "__main__":
    unittest.main()
