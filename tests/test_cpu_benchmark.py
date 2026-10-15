import unittest

import torch

from benchmark_cpu import count_neural_macs, stats
from src.model.modules import GroupedGRNNLayer


class Network(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


class CpuBenchmarkTests(unittest.TestCase):
    def test_latency_units_and_realtime_factor(self):
        result = stats([10, 20, 30], 1)
        self.assertEqual(result["median_ms"], 20)
        self.assertEqual(result["rtf_median"], .02)
        self.assertEqual(result["audio_seconds_per_second"], 50)
        self.assertEqual(result["std_ms"], 10)

    def test_dense_linear_and_grouped_convolution_mac_counts(self):
        cases = [(torch.nn.Linear(4, 5), torch.randn(2, 3, 4), 120),
                 (torch.nn.Conv2d(4, 6, 3, padding=1, groups=2), torch.randn(1, 4, 5, 7), 3780),
                 (torch.nn.ConvTranspose2d(4, 6, 3, groups=2), torch.randn(1, 4, 5, 7), 3780)]
        for layer, x, expected in cases:
            self.assertEqual(count_neural_macs(Network(layer), {"x": x}), expected)

    def test_gru_counts_all_gates_and_layers(self):
        layer = torch.nn.GRU(4, 6, num_layers=2, batch_first=True)
        self.assertEqual(count_neural_macs(Network(layer), {"x": torch.randn(2, 3, 4)}), 2376)

    def test_fastgrnn_counts_shared_gate_projection_groups_and_low_rank(self):
        for bidirectional, rank, expected in [(False, None, 384), (True, None, 768), (False, 1, 192)]:
            layer = GroupedGRNNLayer(8, 8, groups=2, bidirectional=bidirectional,
                                     w_rank=rank, u_rank=rank, backend="torch")
            self.assertEqual(count_neural_macs(Network(layer), {"x": torch.randn(2, 3, 8)}), expected)


if __name__ == "__main__":
    unittest.main()
