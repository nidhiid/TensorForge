"""Drive TensorForge's optimization, CPU lowering, and MLIR JIT runner."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Sequence


@dataclass(frozen=True)
class CompilationResult:
    output_values: tuple[float, ...]
    fusion_occurred: bool
    input_mlir: Path | None
    optimized_mlir: Path | None
    llvm_mlir: Path | None


@dataclass(frozen=True)
class ComparisonResult:
    passed: bool
    max_absolute_error: float


_CPU_LOWERING_PASSES = (
    "--tf-lower-to-linalg",
    "--one-shot-bufferize=bufferize-function-boundaries",
    "--convert-bufferization-to-memref",
    "--convert-linalg-to-loops",
    "--convert-scf-to-cf",
    "--convert-math-to-llvm",
    "--expand-strided-metadata",
    "--convert-arith-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-index-to-llvm",
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm",
    "--reconcile-unrealized-casts",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _find_executable(explicit: str | os.PathLike[str] | None, *candidates: str) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"executable not found: {path}")

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return Path(resolved)
        path = Path(candidate)
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"could not find any of: {', '.join(candidates)}")


def _runner_utils(mlir_runner: Path, explicit: str | os.PathLike[str] | None) -> Path:
    if explicit is not None:
        library = Path(explicit).expanduser().resolve()
        if library.is_file():
            return library
        raise FileNotFoundError(f"MLIR runner utilities not found: {library}")

    library_names = ("libmlir_runner_utils.dylib", "libmlir_runner_utils.so")
    for directory in (mlir_runner.parent.parent / "lib", Path("/opt/homebrew/opt/llvm/lib")):
        for name in library_names:
            library = directory / name
            if library.is_file():
                return library.resolve()
    raise FileNotFoundError("could not locate libmlir_runner_utils")


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(f"command failed: {' '.join(command)}\n{details}") from error


def _parse_memref_output(output: str, expected_elements: int) -> tuple[float, ...]:
    marker = "data ="
    if marker not in output:
        raise RuntimeError(f"MLIR runner did not print a tensor:\n{output}")
    data = output.split(marker, 1)[1]
    tokens = re.findall(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", data
    )
    values = tuple(float(token) for token in tokens)
    if len(values) != expected_elements:
        raise RuntimeError(
            f"expected {expected_elements} output values, runner printed {len(values)}"
        )
    return values


def compile_and_run(
    input_mlir: str,
    output_shape: Sequence[int],
    *,
    artifacts_dir: str | os.PathLike[str] | None = None,
    tensorforge_opt: str | os.PathLike[str] | None = None,
    mlir_runner: str | os.PathLike[str] | None = None,
    runner_utils: str | os.PathLike[str] | None = None,
) -> CompilationResult:
    """Optimize, lower, JIT-execute, and return a printed FP32 tensor."""

    compiler = _find_executable(
        tensorforge_opt, str(_repo_root() / "build/bin/tensorforge-opt")
    )
    runner = _find_executable(
        mlir_runner, "mlir-runner", "/opt/homebrew/opt/llvm/bin/mlir-runner"
    )
    runtime_library = _runner_utils(runner, runner_utils)

    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if artifacts_dir is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="tensorforge-")
        output_directory = Path(temporary_directory.name)
    else:
        output_directory = Path(artifacts_dir).expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

    source_path = output_directory / "input.mlir"
    optimized_path = output_directory / "optimized.mlir"
    llvm_path = output_directory / "lowered-llvm.mlir"
    source_path.write_text(input_mlir, encoding="utf-8")

    _run(
        [
            str(compiler),
            str(source_path),
            "--tf-fuse-linear-gelu",
            "--canonicalize",
            "-o",
            str(optimized_path),
        ]
    )
    optimized = optimized_path.read_text(encoding="utf-8")
    fusion_occurred = "tf.fused_linear_gelu" in optimized

    _run(
        [
            str(compiler),
            str(optimized_path),
            *_CPU_LOWERING_PASSES,
            "-o",
            str(llvm_path),
        ]
    )
    execution = _run(
        [
            str(runner),
            str(llvm_path),
            "-e",
            "main",
            "--entry-point-result=void",
            f"--shared-libs={runtime_library}",
        ]
    )
    values = _parse_memref_output(execution.stdout, math.prod(output_shape))

    result = CompilationResult(
        output_values=values,
        fusion_occurred=fusion_occurred,
        input_mlir=source_path if artifacts_dir is not None else None,
        optimized_mlir=optimized_path if artifacts_dir is not None else None,
        llvm_mlir=llvm_path if artifacts_dir is not None else None,
    )

    # Paths are useful only when the caller requested persistent artifacts.
    if temporary_directory is not None:
        temporary_directory.cleanup()
    return result


def compare_outputs(
    actual: Sequence[float],
    expected: Sequence[float],
    *,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-6,
) -> ComparisonResult:
    if len(actual) != len(expected):
        raise ValueError(
            f"output lengths differ: TensorForge={len(actual)}, PyTorch={len(expected)}"
        )

    max_error = 0.0
    passed = True
    for actual_value, expected_value in zip(actual, expected):
        if not math.isfinite(actual_value) or not math.isfinite(expected_value):
            passed = False
            max_error = math.inf
            continue
        error = abs(actual_value - expected_value)
        max_error = max(max_error, error)
        if error > atol + rtol * abs(expected_value):
            passed = False
    return ComparisonResult(passed=passed, max_absolute_error=max_error)
