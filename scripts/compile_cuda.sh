#!/usr/bin/env bash
# scripts/compile_cuda.sh
set -euo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "CUDA Path: $(which nvcc)"
nvcc --version

cd third_party/llama_cpp_src
rm -rf build
mkdir build
cd build

cmake .. \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release

cmake --build . --config Release -j8
