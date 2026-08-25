#!/usr/bin/env python3
"""Compile and run a tiny PyTorch transformer MLP with TensorForge CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

import torch
from torch import nn

from tensorforge import (
    capture_mlp,
    compare_outputs,
    compile_and_run_cuda,
    emit_mlir,
)


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
        default=REPOSITORY_ROOT / "artifacts/phase10-cuda",
        type=Path,
    )
    parser.add_argument("--cuda-runtime", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is unavailable; run this example on an NVIDIA GPU machine.")
        return 77

    torch.manual_seed(7)
    model = TransformerMLP().eval()
    example_input = torch.randn(1, 2, 4, dtype=torch.float32)
    captured = capture_mlp(model, example_input)

    compiled = compile_and_run_cuda(
        emit_mlir(captured),
        captured.reference_output.shape,
        artifacts_dir=args.artifacts_dir,
        cuda_runtime=args.cuda_runtime,
    )
    comparison = compare_outputs(
        compiled.output_values,
        captured.reference_output.values,
        rtol=2.0e-4,
        atol=2.0e-5,
    )

    print("Backend:           TensorForge CUDA")
    print("Graph:             Linear -> GELU(tanh) -> Linear")
    print(f"GPU:               {torch.cuda.get_device_name(0)}")
    print(f"Fusion occurred:   {'yes' if compiled.fusion_occurred else 'no'}")
    print(f"Maximum error:     {comparison.max_absolute_error:.3e}")
    print(f"Correctness check: {'PASS' if comparison.passed else 'FAIL'}")
    print(f"Artifacts:         {args.artifacts_dir.resolve()}")
    return 0 if comparison.passed and compiled.fusion_occurred else 1


if __name__ == "__main__":
    raise SystemExit(main())
