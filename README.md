# CS79995 Assignment 4: 1-D Convolution in CUDA

This repo has four CUDA versions of a 1-D convolution kernel. The first three each store the mask and input data in a different kind of GPU memory. The fourth is an optimized rewrite of the shared-memory version. Every program also runs a single-threaded CPU version, times both runs and checks that the GPU result matches the CPU result.

| File | Memory strategy | Where the mask `K` lives | Where the input `I` is read from |
| --- | --- | --- | --- |
| [convolution_naive.cu](convolution_naive.cu) | Naive (no memory optimization) | Global memory | Global memory |
| [convolution_constant.cu](convolution_constant.cu) | Constant memory | `__constant__` memory | Global memory |
| [convolution_shared.cu](convolution_shared.cu) | Shared memory + constant memory (tiled) | `__constant__` memory | Shared-memory tile for the block's own elements; global memory for halo elements |
| [convolution_optimized.cu](convolution_optimized.cu) | Optimized tiled shared memory + constant memory | `__constant__` memory | Shared-memory tile for the block's own elements **and** the halo elements; the inner loop never touches global memory |

## How it works

Each GPU thread computes one output element `O[i]`:

```
O[i] = Σ_j  I[i - Mask_Width/2 + j] * K[j]     for 0 ≤ i - Mask_Width/2 + j < Width
```

- **Naive:** reads `I` and `K` from global memory on every multiply.
- **Constant:** copies the mask into `__constant__ float K[MAX_MASK_WIDTH]` with `cudaMemcpyToSymbol`. All threads in a warp read the same `K[j]` at the same time, so the constant cache can broadcast it to the whole warp.
- **Shared:** keeps the mask in constant memory. Each block also loads its part of `I` into a `__shared__` tile and calls `__syncthreads()`. After that, reads that fall inside the tile come from shared memory, and reads outside it (halo cells) go to global memory.
- **Optimized:** keeps the mask in constant memory. Each block loads its part of `I` *plus* `Mask_Width/2` halo cells on each side into a `__shared__` tile, storing `0.0f` for positions outside the input. After `__syncthreads()` the inner loop is a plain multiply-add over shared memory with no bounds check and no global-memory reads. See [Optimized version](#optimized-version) below for the full list of changes.

Each program then:

1. Fills the input and mask arrays with random integers from 0 to 8.
2. Copies the data to the device, launches the kernel and copies the result back.
3. Times the GPU work (kernel plus device-to-host copy) with CUDA events.
4. Runs the same convolution on the CPU and times it.
5. Checks the result. The three original programs compare the sum of the GPU output with the sum of the CPU output. The optimized program compares every element.

## Requirements

- An NVIDIA GPU
- The CUDA Toolkit (`nvcc`) with C++11 or later, because the programs use lambdas and `static_assert`

## Build and run

```bash
nvcc -o convolution_naive     convolution_naive.cu
nvcc -o convolution_constant  convolution_constant.cu
nvcc -o convolution_shared    convolution_shared.cu
nvcc -o convolution_optimized convolution_optimized.cu

./convolution_naive
./convolution_constant
./convolution_shared
./convolution_optimized
```

Example output of the three original programs (the timings depend on your hardware):

```
Using Device 0: <GPU name>
size of i array: 2000
size of k array: 500
size of block: 10
Compute time on GPU: <t> ms
Compute time on CPU: <t> ms
Check Result: Arrays matched
```

Example output of the optimized program:

```
Using Device 0: <GPU name>
size of i array: 2000
size of k array: 500
size of o array: 2499
mask width: 401 (radius 200)
threads per block: 256
number of blocks: 10
shared tile per block: 656 floats (2624 bytes)
Kernel time on GPU: <t> ms
Compute time on GPU: <t> ms  (kernel + device-to-host copy)
Compute time on CPU: <t> ms
Check Result: Arrays matched
```

## Optimized version

`convolution_optimized.cu` solves exactly the same problem as `convolution_shared.cu`: same `ISIZE`, `KSIZE`, `Width` and `Mask_Width`, same random data range and the same output values. Only the GPU implementation and the measurement code change. The original three files are untouched.

### Kernel differences

| Aspect | `convolution_shared.cu` (original) | `convolution_optimized.cu` |
| --- | --- | --- |
| What goes into the shared tile | Only the block's own `blockDim.x` input elements (`I_ds[threadIdx.x] = I[i]`) | The block's elements plus `Mask_Width/2` halo cells on each side (`TILE_SIZE = THREADS_PER_BLOCK + Mask_Width - 1`) |
| How the tile is loaded | One element per thread | Cooperative strided loop, each thread loads `ceil(TILE_SIZE / blockDim.x)` consecutive-stride elements, so global reads stay coalesced |
| Halo cells | Read from global memory inside the inner loop | Read once into shared memory during the load phase |
| Out-of-range inputs | Skipped with `if (I_index >= 0 && I_index < Width)` inside the inner loop | Stored as `0.0f` in the tile during the load phase; the inner loop has no bounds check |
| Inner loop body | Two nested `if` statements choosing between a shared-memory read and a global-memory read, then a multiply-add | One multiply-add: `Ovalue += I_ds[tx + j] * K[j]` |
| Warp divergence in the inner loop | Yes: with `Mask_Width/2 = 200` and 200 threads per block, half or more of the reads fall outside the block's own tile, and threads in the same warp take different branches | None: every thread runs the same straight-line code |
| Global memory traffic in the inner loop | Every halo read is a separate global load (cached in L1/L2 at best) | Zero |
| Loop unrolling | None | `#pragma unroll 8` with a compile-time loop bound (`MASK_WIDTH`) |
| Pointer qualifiers | None | `const float * __restrict__ I`, `float * __restrict__ O` |
| Shared-memory access pattern | `I_ds[threadIdx.x + j - Mask_Width/2]` | `I_ds[tx + j]`: consecutive threads read consecutive words, no bank conflicts |

### Launch configuration

| | `convolution_shared.cu` | `convolution_optimized.cu` |
| --- | --- | --- |
| Threads per block | `ISIZE / BLOCK_SIZE` = 200 (not a multiple of the 32-thread warp: the seventh warp of every block has only 8 of its 32 lanes active) | `THREADS_PER_BLOCK` = 256 (a multiple of the warp size) |
| Blocks | `BLOCK_SIZE` = 10 | `ceil(OSIZE / THREADS_PER_BLOCK)` = 10 for the default sizes |
| Outputs computed | The first `ISIZE` = 2000 of the 2499 outputs; the last 499 are never written | All `OSIZE` = 2499 outputs, guarded by `if (i < Osize)` |
| Shared memory per block | 500 floats (2000 bytes), of which only 200 are used | 656 floats (2624 bytes), all used |

### Host-side differences

- **Error checking.** Every CUDA call goes through a `CHECK` macro that prints `cudaGetErrorString` and exits. The original files only check `cudaGetDeviceProperties` and `cudaSetDevice`.
- **Compile-time checks.** `static_assert` confirms that the mask fits in the host buffer and in constant memory, that the block size is a multiple of 32 and that the tile fits in the 48 KB static shared-memory limit.
- **Warm-up launch.** The kernel is launched once before timing. The first launch in a process pays for CUDA context creation and module loading, which the original files count as GPU compute time.
- **Separate timings.** The program prints the kernel time alone and the kernel plus device-to-host copy time (the latter is what the original files measure). The CPU run is timed with `std::chrono::steady_clock` instead of CUDA events.
- **Element-wise correctness check.** The CPU reference produces a full output array and every element is compared with a relative tolerance of `1e-5`. The first five mismatches, if any, are printed and the program exits with a non-zero status. The original files only compare the sums of the two outputs, stored in `int` variables, so two outputs could differ and still "match".
- **Only one mask copy.** The optimized file does not allocate the unused `dev_k` global-memory buffer that the constant and shared versions still create.

### Why it is faster

For the default sizes each output needs 401 multiply-adds. In the original shared version, a thread with `threadIdx.x = t` reads tile positions `t - 200 ... t + 200`, but the tile only covers positions `0 ... 199`. So about half of the 401 reads in every thread leave the tile and go to global memory, and the two `if` statements run on every iteration. The optimized kernel does all the global loading up front in a coalesced pass of at most three loads per thread, then runs 401 iterations that read only shared memory and constant memory with no branches. The removed branches also let `nvcc` unroll the loop and schedule the multiply-adds back to back.

### Verification

The kernel's indexing was checked with a CPU emulation of the block/thread structure (load phase for all threads, then compute phase) against the same reference formula the original programs use. It matched element for element for `ISIZE` values of 1000, 1500 and 2000 and block sizes of 32, 64, 128 and 256 threads, including the zero-padded halos and the partially filled last block. The file was written on a machine without `nvcc`, so build it with the command above before relying on the timings.

## Configuration

The problem sizes are set with `#define` at the top of each file:

| Macro | Meaning | naive | constant | shared | optimized |
| --- | --- | --- | --- | --- | --- |
| `ISIZE` | Input array length | 2000 | 1500 | 2000 | 2000 |
| `KSIZE` | Mask array length | 500 | 500 | 500 | 500 |
| `BLOCK_SIZE` | Number of blocks in the grid | 10 | 10 | 10 | — (derived from `OSIZE`) |
| `THREADS_PER_BLOCK` | Threads per block | — | — | — | 256 |
| `MAX_MASK_WIDTH` | Size of the constant-memory mask | — | 500 | 500 | 500 |
| `TILE_SIZE` | Size of the shared-memory tile | — | — | 500 | `THREADS_PER_BLOCK + MASK_WIDTH - 1` = 656 |

In the three original files the kernel is launched as `<<<BLOCK_SIZE, ISIZE / BLOCK_SIZE>>>`, so the grid has `BLOCK_SIZE` blocks and each block has `ISIZE / BLOCK_SIZE` threads. The optimized file is launched as `<<<ceil(OSIZE / THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>`. In every version the effective convolution parameters are derived from `ISIZE`:

- `Width = ISIZE / 10`
- `Mask_Width = 2 * Width + 1`

In `convolution_shared.cu`, `KSIZE`, `TILE_SIZE` and `MAX_MASK_WIDTH` should all be the same value. `TILE_SIZE` must also be at least the number of threads per block.

In `convolution_optimized.cu`, `WIDTH`, `MASK_WIDTH`, `OSIZE` and `TILE_SIZE` are all derived automatically from `ISIZE`, `KSIZE` and `THREADS_PER_BLOCK`. To change the problem size, edit only those three macros; the `static_assert` lines will stop the build if the mask no longer fits.

## Limitations

These apply to the three original files:

- Because `Width = ISIZE / 10`, the kernel only reads the first `ISIZE / 10` input elements, not the whole input array.
- The launch creates `ISIZE` threads, so only the first `ISIZE` of the `ISIZE + KSIZE - 1` output elements are computed.
- The kernels have no bounds check on `i`, so the thread count must not be larger than the output buffer.
- The correctness check compares the sums of the two outputs, not each element. It also stores those sums in `int` variables.
- The GPU time includes the device-to-host copy but not the host-to-device copy.

`convolution_optimized.cu` fixes the second, third and fourth points and reports the kernel time separately from the copy. It deliberately keeps `Width = ISIZE / 10` so its results and timings can be compared directly with the original versions.
