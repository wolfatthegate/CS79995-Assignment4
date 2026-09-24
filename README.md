# CS79995 Assignment 4: 1-D Convolution in CUDA

This repo has three CUDA versions of a 1-D convolution kernel. Each one stores the mask and input data in a different kind of GPU memory. Every program also runs a single-threaded CPU version, times both runs and checks that the GPU result matches the CPU result.

| File | Memory strategy | Where the mask `K` lives | Where the input `I` is read from |
| --- | --- | --- | --- |
| [convolution_naive.cu](convolution_naive.cu) | Naive (no memory optimization) | Global memory | Global memory |
| [convolution_constant.cu](convolution_constant.cu) | Constant memory | `__constant__` memory | Global memory |
| [convolution_shared.cu](convolution_shared.cu) | Shared memory + constant memory (tiled) | `__constant__` memory | Shared-memory tile for the block's own elements; global memory for halo elements |

## How it works

Each GPU thread computes one output element `O[i]`:

```
O[i] = Σ_j  I[i - Mask_Width/2 + j] * K[j]     for 0 ≤ i - Mask_Width/2 + j < Width
```

- **Naive:** reads `I` and `K` from global memory on every multiply.
- **Constant:** copies the mask into `__constant__ float K[MAX_MASK_WIDTH]` with `cudaMemcpyToSymbol`. All threads in a warp read the same `K[j]` at the same time, so the constant cache can broadcast it to the whole warp.
- **Shared:** keeps the mask in constant memory. Each block also loads its part of `I` into a `__shared__` tile and calls `__syncthreads()`. After that, reads that fall inside the tile come from shared memory, and reads outside it (halo cells) go to global memory.

Each program then:

1. Fills the input and mask arrays with random integers from 0 to 8.
2. Copies the data to the device, launches the kernel and copies the result back.
3. Times the GPU work (kernel plus device-to-host copy) with CUDA events.
4. Runs the same convolution on the CPU and times it.
5. Checks the result by comparing the sum of the GPU output with the sum of the CPU output.

## Requirements

- An NVIDIA GPU
- The CUDA Toolkit (`nvcc`) with C++11 or later, because the programs use lambdas

## Build and run

```bash
nvcc -o convolution_naive    convolution_naive.cu
nvcc -o convolution_constant convolution_constant.cu
nvcc -o convolution_shared   convolution_shared.cu

./convolution_naive
./convolution_constant
./convolution_shared
```

Example output (the timings depend on your hardware):

```
Using Device 0: <GPU name>
size of i array: 2000
size of k array: 500
size of block: 10
Compute time on GPU: <t> ms
Compute time on CPU: <t> ms
Check Result: Arrays matched
```

## Configuration

The problem sizes are set with `#define` at the top of each file:

| Macro | Meaning | naive | constant | shared |
| --- | --- | --- | --- | --- |
| `ISIZE` | Input array length | 2000 | 1500 | 2000 |
| `KSIZE` | Mask array length | 500 | 500 | 500 |
| `BLOCK_SIZE` | Number of blocks in the grid | 10 | 10 | 10 |
| `MAX_MASK_WIDTH` | Size of the constant-memory mask | — | 500 | 500 |
| `TILE_SIZE` | Size of the shared-memory tile | — | — | 500 |

The kernel is launched as `<<<BLOCK_SIZE, ISIZE / BLOCK_SIZE>>>`, so the grid has `BLOCK_SIZE` blocks and each block has `ISIZE / BLOCK_SIZE` threads. The effective convolution parameters are derived from `ISIZE`:

- `Width = ISIZE / 10`
- `Mask_Width = 2 * Width + 1`

In `convolution_shared.cu`, `KSIZE`, `TILE_SIZE` and `MAX_MASK_WIDTH` should all be the same value. `TILE_SIZE` must also be at least the number of threads per block.

## Limitations

- Because `Width = ISIZE / 10`, the kernel only reads the first `ISIZE / 10` input elements, not the whole input array.
- The launch creates `ISIZE` threads, so only the first `ISIZE` of the `ISIZE + KSIZE - 1` output elements are computed.
- The kernels have no bounds check on `i`, so the thread count must not be larger than the output buffer.
- The correctness check compares the sums of the two outputs, not each element. It also stores those sums in `int` variables.
- The GPU time includes the device-to-host copy but not the host-to-device copy.
