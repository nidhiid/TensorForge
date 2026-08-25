#!/usr/bin/env python3
"""Benchmark TensorForge fused versus unfused CUDA Linear+GELU kernels."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as functional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUCCESS = 0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


class CudaRuntime:
    def __init__(self, library_path: Path) -> None:
        self.library = ctypes.CDLL(str(library_path.resolve()))
        self.context = ctypes.c_void_p()

        self.library.tfCudaCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int32,
        ]
        self.library.tfCudaCreate.restype = ctypes.c_int32
        self.library.tfCudaDestroy.argtypes = [ctypes.c_void_p]
        self.library.tfCudaDestroy.restype = ctypes.c_int32
        self.library.tfCudaMalloc.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.library.tfCudaMalloc.restype = ctypes.c_int32
        self.library.tfCudaFree.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.library.tfCudaFree.restype = ctypes.c_int32
        self.library.tfCudaCopyHostToDevice.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.library.tfCudaCopyHostToDevice.restype = ctypes.c_int32
        self.library.tfCudaCopyDeviceToHost.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.library.tfCudaCopyDeviceToHost.restype = ctypes.c_int32
        self.library.tfCudaSynchronize.argtypes = [ctypes.c_void_p]
        self.library.tfCudaSynchronize.restype = ctypes.c_int32
        self.library.tfCudaFusedLinearGeluF32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self.library.tfCudaFusedLinearGeluF32.restype = ctypes.c_int32
        self.library.tfCudaTimeLinearGeluF32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.library.tfCudaTimeLinearGeluF32.restype = ctypes.c_int32
        self.library.tfCudaGetLastError.argtypes = [ctypes.c_void_p]
        self.library.tfCudaGetLastError.restype = ctypes.c_char_p

        self._check(self.library.tfCudaCreate(ctypes.byref(self.context), 0))

    def _check(self, status: int) -> None:
        if status == SUCCESS:
            return
        message = self.library.tfCudaGetLastError(self.context)
        detail = message.decode() if message else "unknown CUDA error"
        raise RuntimeError(f"TensorForge CUDA failed with status {status}: {detail}")

    def allocate(self, bytes_count: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self._check(
            self.library.tfCudaMalloc(
                self.context, ctypes.byref(pointer), bytes_count
            )
        )
        return pointer

    def upload(self, destination: ctypes.c_void_p, tensor: torch.Tensor) -> None:
        self._check(
            self.library.tfCudaCopyHostToDevice(
                self.context,
                destination,
                ctypes.c_void_p(tensor.data_ptr()),
                tensor.numel() * tensor.element_size(),
            )
        )

    def download(self, tensor: torch.Tensor, source: ctypes.c_void_p) -> None:
        self._check(
            self.library.tfCudaCopyDeviceToHost(
                self.context,
                ctypes.c_void_p(tensor.data_ptr()),
                source,
                tensor.numel() * tensor.element_size(),
            )
        )
        self._check(self.library.tfCudaSynchronize(self.context))

    def close(self, allocations: list[ctypes.c_void_p]) -> None:
        for pointer in reversed(allocations):
            self._check(self.library.tfCudaFree(self.context, pointer))
        self._check(self.library.tfCudaDestroy(self.context))
        self.context = ctypes.c_void_p()


def benchmark_workload(
    runtime: CudaRuntime,
    workload: dict[str, int | str],
    warmup: int,
    samples: int,
    iterations: int,
) -> dict[str, float | int | str]:
    m = int(workload["batch"]) * int(workload["sequence"])
    k = int(workload["hidden"])
    n = int(workload["intermediate"])
    generator = torch.Generator().manual_seed(2026)
    input_tensor = torch.randn(m, k, generator=generator, dtype=torch.float32)
    weight = torch.randn(k, n, generator=generator, dtype=torch.float32) * 0.02
    bias = torch.randn(n, generator=generator, dtype=torch.float32) * 0.02
    output = torch.empty(m, n, dtype=torch.float32)

    allocations: list[ctypes.c_void_p] = []
    try:
        input_device = runtime.allocate(input_tensor.nbytes)
        allocations.append(input_device)
        weight_device = runtime.allocate(weight.nbytes)
        allocations.append(weight_device)
        bias_device = runtime.allocate(bias.nbytes)
        allocations.append(bias_device)
        scratch_device = runtime.allocate(output.nbytes)
        allocations.append(scratch_device)
        output_device = runtime.allocate(output.nbytes)
        allocations.append(output_device)
        runtime.upload(input_device, input_tensor)
        runtime.upload(weight_device, weight)
        runtime.upload(bias_device, bias)

        runtime._check(
            runtime.library.tfCudaFusedLinearGeluF32(
                runtime.context,
                input_device,
                weight_device,
                bias_device,
                output_device,
                m,
                k,
                n,
            )
        )
        runtime.download(output, output_device)
        input_torch = input_tensor.cuda()
        weight_torch = weight.cuda()
        bias_torch = bias.cuda()
        reference = functional.gelu(
            input_torch @ weight_torch + bias_torch, approximate="tanh"
        ).cpu()
        torch.testing.assert_close(output, reference, rtol=2.0e-4, atol=2.0e-5)

        def time(fused: bool, count: int) -> float:
            elapsed = ctypes.c_float()
            runtime._check(
                runtime.library.tfCudaTimeLinearGeluF32(
                    runtime.context,
                    input_device,
                    weight_device,
                    bias_device,
                    scratch_device,
                    output_device,
                    m,
                    k,
                    n,
                    int(fused),
                    count,
                    ctypes.byref(elapsed),
                )
            )
            return elapsed.value

        time(False, warmup)
        time(True, warmup)
        unfused_times = [time(False, iterations) for _ in range(samples)]
        fused_times = [time(True, iterations) for _ in range(samples)]

        def time_pytorch(count: int) -> float:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(count):
                functional.gelu(
                    input_torch @ weight_torch + bias_torch,
                    approximate="tanh",
                )
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / count

        time_pytorch(warmup)
        pytorch_times = [time_pytorch(iterations) for _ in range(samples)]
        unfused_ms = statistics.median(unfused_times)
        fused_ms = statistics.median(fused_times)

        # The unfused path writes the Linear result and reads it for GELU.
        avoided_bytes = 2 * m * n * 4
        baseline_bytes = (m * k + k * n + n + 3 * m * n) * 4
        return {
            "workload": str(workload["name"]),
            "m": m,
            "k": k,
            "n": n,
            "unfused_ms": unfused_ms,
            "unfused_p95_ms": percentile(unfused_times, 0.95),
            "fused_ms": fused_ms,
            "fused_p95_ms": percentile(fused_times, 0.95),
            "pytorch_eager_ms": statistics.median(pytorch_times),
            "pytorch_eager_p95_ms": percentile(pytorch_times, 0.95),
            "speedup": unfused_ms / fused_ms,
            "theoretical_dram_reduction_percent": 100.0
            * avoided_bytes
            / baseline_bytes,
        }
    finally:
        runtime.close(allocations)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=REPOSITORY_ROOT / "build-cuda/lib/libTensorForgeCudaRuntime.so",
    )
    parser.add_argument("--workload", default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is unavailable; run this benchmark on an NVIDIA GPU machine.")
        return 77
    if min(args.warmup, args.samples, args.iterations) <= 0:
        parser.error("warmup, samples, and iterations must all be positive")

    configuration = json.loads(
        (REPOSITORY_ROOT / "config/workloads.json").read_text(encoding="utf-8")
    )
    workloads = configuration["workloads"]
    if args.workload != "all":
        workloads = [item for item in workloads if item["name"] == args.workload]
        if not workloads:
            parser.error(f"unknown workload: {args.workload}")

    results = []
    for workload in workloads:
        runtime = CudaRuntime(args.runtime)
        result = benchmark_workload(
            runtime, workload, args.warmup, args.samples, args.iterations
        )
        results.append(result)
        print(
            f"{result['workload']:12} [{result['m']}x{result['k']}] "
            f"x [{result['k']}x{result['n']}]: "
            f"{result['unfused_ms']:.3f} ms -> {result['fused_ms']:.3f} ms, "
            f"{result['speedup']:.2f}x; PyTorch={result['pytorch_eager_ms']:.3f} ms; "
            "correctness PASS"
        )

    mean_speedup = statistics.fmean(float(item["speedup"]) for item in results)
    mean_traffic = statistics.fmean(
        float(item["theoretical_dram_reduction_percent"]) for item in results
    )
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "results": results,
        "average_speedup": mean_speedup,
        "average_theoretical_dram_reduction_percent": mean_traffic,
    }
    print(f"Average custom-kernel speedup: {mean_speedup:.2f}x")
    print(f"Theoretical DRAM traffic reduction: {mean_traffic:.1f}%")
    print("Note: use Nsight Compute to replace the theoretical traffic estimate.")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON report: {args.json_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
