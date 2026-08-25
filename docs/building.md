# Building TensorForge

Phase 2 requires a C++17 compiler, CMake, Ninja, and an LLVM installation that
includes MLIR development files. LLVM and MLIR must come from the same build or
package version.

## macOS with Homebrew

Install the dependencies:

```bash
brew install cmake ninja llvm
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
```

It should parse the file and print the same MLIR module. Use `-o output.mlir`
to write the result to another file and `--help` to see MLIR's standard driver
options.
