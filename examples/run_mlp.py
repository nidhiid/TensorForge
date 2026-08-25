#!/usr/bin/env python3
"""Run a tiny PyTorch transformer MLP through TensorForge on the CPU."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

import torch
from torch import nn

from tensorforge import capture_mlp, compile_and_run, compare_outputs, emit_mlir


class TransformerMLP(nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 8) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden, intermediate)
        self.gelu = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(intermediate, hidden)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.gelu(self.linear1(value)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        default=REPOSITORY_ROOT / "artifacts/phase7",
        type=Path,
        help="directory for input, optimized, and LLVM MLIR files",
    )
    args = parser.parse_args()

    torch.manual_seed(7)
    model = TransformerMLP().eval()
    example_input = torch.randn(1, 2, 4, dtype=torch.float32)

    captured = capture_mlp(model, example_input)
    input_mlir = emit_mlir(captured)
    compiled = compile_and_run(
        input_mlir,
        captured.reference_output.shape,
        artifacts_dir=args.artifacts_dir,
    )
    comparison = compare_outputs(
        compiled.output_values, captured.reference_output.values
    )

    print("Captured graph: Linear -> GELU(tanh) -> Linear")
    print(f"Input shape:       {captured.original_input_shape}")
    print(f"Fusion occurred:   {'yes' if compiled.fusion_occurred else 'no'}")
    print(f"Output shape:      {captured.reference_output.shape}")
    print(f"Maximum error:     {comparison.max_absolute_error:.3e}")
    print(f"Correctness check: {'PASS' if comparison.passed else 'FAIL'}")
    print(f"Artifacts:         {args.artifacts_dir.resolve()}")
    return 0 if comparison.passed and compiled.fusion_occurred else 1


if __name__ == "__main__":
    raise SystemExit(main())
