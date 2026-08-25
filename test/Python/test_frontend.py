from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from tensorforge import (
    FrontendError,
    capture_mlp,
    compile_and_run,
    compare_outputs,
    emit_mlir,
)


class SupportedMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(2, 3)
        self.gelu = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.gelu(self.linear1(value)))


class FrontendTests(unittest.TestCase):
    def test_capture_transposes_weights_and_emits_static_reshape(self) -> None:
        model = SupportedMLP().eval()
        with torch.no_grad():
            model.linear1.weight.copy_(
                torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )
        input_value = torch.ones((1, 2, 2), dtype=torch.float32)

        captured = capture_mlp(model, input_value)
        mlir = emit_mlir(captured)

        self.assertEqual(captured.first_linear.weight.shape, (2, 3))
        self.assertEqual(
            captured.first_linear.weight.values,
            (1.0, 3.0, 5.0, 2.0, 4.0, 6.0),
        )
        self.assertIn("tensor<2x3xf32>", mlir)
        self.assertIn("tf.reshape %linear2 to [1, 2, 2]", mlir)
        self.assertIn("bufferization.to_buffer", mlir)

    def test_rejects_unsupported_activation(self) -> None:
        model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2))
        with self.assertRaisesRegex(FrontendError, "expected GELU"):
            capture_mlp(model, torch.ones((1, 2), dtype=torch.float32))

    def test_rejects_exact_gelu(self) -> None:
        model = nn.Sequential(nn.Linear(2, 3), nn.GELU(), nn.Linear(3, 2))
        with self.assertRaisesRegex(FrontendError, "approximate='tanh'"):
            capture_mlp(model, torch.ones((1, 2), dtype=torch.float32))

    def test_non_finite_output_fails_comparison(self) -> None:
        comparison = compare_outputs([float("nan")], [0.0])
        self.assertFalse(comparison.passed)
        self.assertEqual(comparison.max_absolute_error, float("inf"))

    def test_pytorch_to_cpu_matches_numerically(self) -> None:
        torch.manual_seed(17)
        captured = capture_mlp(
            SupportedMLP().eval(), torch.randn((1, 2, 2), dtype=torch.float32)
        )

        with tempfile.TemporaryDirectory(prefix="tensorforge-python-test-") as temp:
            compiled = compile_and_run(
                emit_mlir(captured),
                captured.reference_output.shape,
                artifacts_dir=temp,
                tensorforge_opt=os.environ.get("TENSORFORGE_OPT"),
                mlir_runner=os.environ.get("MLIR_RUNNER"),
                runner_utils=os.environ.get("MLIR_RUNNER_UTILS"),
            )
            comparison = compare_outputs(
                compiled.output_values, captured.reference_output.values
            )

            self.assertTrue(compiled.fusion_occurred)
            self.assertTrue(comparison.passed)
            self.assertLess(comparison.max_absolute_error, 1.0e-5)
            self.assertIn(
                "tf.fused_linear_gelu",
                Path(compiled.optimized_mlir).read_text(encoding="utf-8"),
            )
            self.assertIn(
                "llvm.func @main",
                Path(compiled.llvm_mlir).read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
