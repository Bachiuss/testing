# Vector Addition (CUDA)

A simple CUDA C++ implementation of vector addition utilizing a grid-stride loop pattern and dynamic SM-based block scheduling.

## Features
- **Grid-Stride Loop (`vecAdd`)**: Allows processing large arrays regardless of the grid size configuration.
- **Hardware-Aware Kernel Launch**: Dynamically queries the GPU's Streaming Multiprocessors (SMs) using `cudaDeviceGetAttribute` to optimize block count (`numSMs * 4`).
- **Error Handling**: Uses custom `CUDA_CHECK` wrappers for runtime debugging.
- **Verification**: Verifies GPU results against CPU calculations with error tolerance checking.

## Usage

### Compilation
nvcc -o vecadd vecadd.cu

### Execution
./vecadd
