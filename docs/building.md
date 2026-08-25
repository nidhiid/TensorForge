# Building TensorForge

Phase 2 requires a C++17 compiler, CMake, Ninja, and an LLVM installation that
includes MLIR development files. LLVM and MLIR must come from the same build or
package version.

## macOS with Homebrew

Install the dependencies:

```bash
brew install cmake ninja llvm
```

Create the Python 3.11 environment and install the PyTorch frontend:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Configure, build, and test:

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR="$(brew --prefix llvm)/lib/cmake/mlir" \
  -DLLVM_DIR="$(brew --prefix llvm)/lib/cmake/llvm"
cmake --build build
ctest --test-dir build --output-on-failure
```

Homebrew keeps LLVM keg-only, so passing `MLIR_DIR` and `LLVM_DIR` explicitly
prevents CMake from accidentally finding Apple Clang's system files.

## LLVM/MLIR built from source

Point CMake at the package configuration directories created by the LLVM
build. For a build directory stored in `$LLVM_BUILD_DIR`, run:

```bash
cmake -S . -B build -G Ninja \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm"
cmake --build build
ctest --test-dir build --output-on-failure
```

## Run the tool

The tool accepts ordinary MLIR containing the registered `func` dialect and
TensorForge's `tf` dialect:

```bash
./build/bin/tensorforge-opt test/smoke.mlir
./build/bin/tensorforge-opt test/Dialect/valid.mlir
./build/bin/tensorforge-opt --canonicalize test/Transforms/canonicalize.mlir
./build/bin/tensorforge-opt --tf-fuse-linear-gelu \
  test/Transforms/fuse-linear-gelu.mlir
./build/bin/tensorforge-opt --tf-fuse-linear-gelu --tf-lower-to-linalg \
  test/Transforms/fuse-linear-gelu.mlir
```

It should parse the file and print the same MLIR module. Use `-o output.mlir`
to write the result to another file and `--help` to see MLIR's standard driver
options.

The test suite also compiles a small fused Linear+GELU graph all the way to
LLVM dialect IR and executes it with MLIR's CPU JIT runner:

```bash
ctest --test-dir build -R cpu --output-on-failure
```

## Run the PyTorch frontend

After building `tensorforge-opt`, run the complete Phase 7 demonstration:

```bash
.venv/bin/python examples/run_mlp.py
```

The command captures a small transformer MLP with `torch.fx`, emits input and
optimized MLIR, runs the native CPU result, and compares every output value
against PyTorch. Compiler artifacts are saved under `artifacts/phase7/`.

## Build the CUDA runtime

The default build sets `TENSORFORGE_ENABLE_CUDA=OFF`. It produces a portable
runtime library whose API reports that CUDA is unavailable, allowing compiler
and API development on machines such as Apple Silicon Macs.

On a Linux machine with the NVIDIA CUDA Toolkit, enable the real runtime and
select the compute capability for the target GPU. For example, architecture
`80` targets NVIDIA Ampere A100 GPUs:

```bash
cmake -S . -B build-cuda -G Ninja \
  -DMLIR_DIR=/path/to/llvm/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DTENSORFORGE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-cuda
ctest --test-dir build-cuda -R cuda --output-on-failure
```

Configuration fails immediately if CUDA is requested but `nvcc` or the CUDA
Toolkit cannot be found. The GPU correctness test returns CTest's skipped
status when the toolkit is installed but no NVIDIA device is visible.

The CUDA test selection includes the Phase 8 MatMul test and the Phase 10
fused/unfused Linear+GELU test. Run the complete compiler-generated CUDA path
and benchmark after the tests pass:

```bash
.venv/bin/python examples/run_mlp_cuda.py \
  --cuda-runtime build-cuda/lib/libTensorForgeCudaRuntime.so
.venv/bin/python examples/benchmark_cuda.py \
  --runtime build-cuda/lib/libTensorForgeCudaRuntime.so \
  --warmup 10 --samples 20 --iterations 100 \
  --json-output artifacts/phase10/benchmark.json
```

The benchmark's fused/unfused timing excludes allocation and transfers. The
current compiler-generated host wrappers do include those costs because they
create device storage for each operation; this distinction should be stated
when presenting results.

## Google Colab kernel-only check

Colab normally supplies CUDA and PyTorch, but not an LLVM/MLIR development
build. You can still compile and verify the runtime kernels. First make sure the
notebook is inside the cloned repository—the paths below are relative to its
root:

```bash
cd /content/TensorForge
pwd
test -f lib/Runtime/CudaRuntime.cpp
```

Then build the shared library and the fused correctness test:

```bash
nvcc -std=c++17 -shared -Xcompiler=-fPIC \
  -Wno-deprecated-gpu-targets \
  -Iinclude \
  lib/Runtime/CudaRuntime.cpp \
  lib/Runtime/MatMul.cu \
  -o libTensorForgeCudaRuntime.so

g++ -std=c++17 -Iinclude \
  test/Runtime/cuda-linear-gelu-correctness.cpp \
  -L. -lTensorForgeCudaRuntime -Wl,-rpath,. \
  -o cuda-linear-gelu-test

./cuda-linear-gelu-test
```

A successful run prints a `PASS` line with fused and unfused milliseconds. The
deprecated-GPU-target message is only an `nvcc` warning; a missing source file
means the notebook is in the wrong directory or has not pulled the latest
changes.
