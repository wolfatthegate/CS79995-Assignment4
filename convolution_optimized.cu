/**
* This program computes the 1-D convolution in parallel with an optimized
* tiled kernel.  It solves the same problem as convolution_shared.cu
* (same input sizes, same Width / Mask_Width derivation, same random data)
* but changes how the GPU does the work:
*
*   1. The whole input tile INCLUDING the halo (ghost) cells is loaded into
*      shared memory once per block.  The inner loop never touches global
*      memory and never has to decide between a shared and a global read.
*   2. Out-of-range input positions are stored in the tile as 0.0f, so the
*      inner loop has no bounds check and no warp divergence.  Multiplying
*      by zero gives the same result as skipping the element.
*   3. The mask stays in __constant__ memory (broadcast through the
*      constant cache, every thread in a warp reads the same K[j]).
*   4. The block size is a multiple of the warp size (256 threads) and the
*      grid covers every output element, so the whole output buffer is
*      computed and the correctness check is meaningful.
*   5. The inner loop is unrolled, the pointers are __restrict__-qualified
*      and the loop bound is a compile-time constant.
*   6. Every CUDA call is error checked, a warm-up launch is done before
*      timing, the kernel is timed separately from the copy, the CPU is
*      timed with std::chrono and the result is compared element by element.
*/
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>

using std::generate;
using std::vector;

// ---------------------------------------------------------------------------
// Problem configuration (same values as convolution_shared.cu)
// ---------------------------------------------------------------------------
#define ISIZE 2000                      // length of input array I
#define KSIZE 500                       // length of mask array K (host buffer)
#define THREADS_PER_BLOCK 256           // multiple of the warp size (32)

// Derived convolution parameters, kept identical to the other versions so
// the results and timings can be compared directly.
#define WIDTH       (ISIZE / 10)        // number of input elements actually read
#define MASK_WIDTH  (2 * WIDTH + 1)     // odd mask width
#define MASK_RADIUS (MASK_WIDTH / 2)
#define OSIZE       (ISIZE + KSIZE - 1) // length of the output array

// Each block needs its own THREADS_PER_BLOCK elements plus MASK_RADIUS
// halo elements on each side.
#define TILE_SIZE   (THREADS_PER_BLOCK + MASK_WIDTH - 1)

#define MAX_MASK_WIDTH 500
__constant__ float K[MAX_MASK_WIDTH];

static_assert(MASK_WIDTH <= KSIZE,          "mask does not fit in the host mask buffer");
static_assert(MASK_WIDTH <= MAX_MASK_WIDTH, "mask does not fit in constant memory");
static_assert(THREADS_PER_BLOCK % 32 == 0,  "block size should be a multiple of the warp size");
static_assert(TILE_SIZE * sizeof(float) <= 48 * 1024, "tile exceeds the static shared-memory limit");

#define CHECK(x) do { cudaError_t err__ = (x); if (err__ != cudaSuccess) { \
	printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
	return EXIT_FAILURE; } } while (0)

// ---------------------------------------------------------------------------
// Optimized tiled kernel
// ---------------------------------------------------------------------------
__global__ void convolution_1D_tiled_kernel(const float * __restrict__ I,
                                            float * __restrict__ O,
                                            int Width, int Osize)
{
	__shared__ float I_ds[TILE_SIZE];

	const int tx          = threadIdx.x;
	const int block_start = blockIdx.x * blockDim.x;      // first output of this block
	const int tile_start  = block_start - MASK_RADIUS;    // first input element in the tile

	// Phase 1: cooperatively load the tile (block elements + both halos).
	// Consecutive threads load consecutive addresses, so the global reads
	// are coalesced.  Positions outside [0, Width) become 0.0f, which is
	// the same as skipping them in the sum.
	for (int idx = tx; idx < TILE_SIZE; idx += blockDim.x) {
		int g = tile_start + idx;
		I_ds[idx] = (g >= 0 && g < Width) ? I[g] : 0.0f;
	}
	__syncthreads();

	// Phase 2: every thread computes one output from shared memory only.
	// Thread tx needs inputs [i - MASK_RADIUS, i + MASK_RADIUS], which sit
	// at tile positions [tx, tx + MASK_WIDTH).  Consecutive threads read
	// consecutive shared-memory words, so there are no bank conflicts, and
	// K[j] is the same for the whole warp, so it is broadcast from the
	// constant cache.
	float Ovalue = 0.0f;
	#pragma unroll 8
	for (int j = 0; j < MASK_WIDTH; j++) {
		Ovalue += I_ds[tx + j] * K[j];
	}

	const int i = block_start + tx;
	if (i < Osize) {
		O[i] = Ovalue;
	}
}

// ---------------------------------------------------------------------------
// CPU reference (same math as the other versions, but it keeps every
// output element so the GPU result can be checked element by element)
// ---------------------------------------------------------------------------
void convolution_1D_CPU(const vector<float> &host_i, const vector<float> &host_k,
                        vector<float> &cpu_o, int Mask_Width, int Width)
{
	for (size_t i = 0; i < cpu_o.size(); i++) {
		float acc = 0.0f;
		int N_start_point = (int)i - (Mask_Width / 2);
		for (int j = 0; j < Mask_Width; j++) {
			int idx = N_start_point + j;
			if (idx >= 0 && idx < Width) {
				acc += host_i[idx] * host_k[j];
			}
		}
		cpu_o[i] = acc;
	}
}

// Compare two arrays element by element with a relative tolerance.
int countMismatches(const vector<float> &a, const vector<float> &b)
{
	int mismatches = 0;
	for (size_t i = 0; i < a.size(); i++) {
		float tol = 1.0e-5f * std::max(1.0f, fabsf(b[i]));
		if (fabsf(a[i] - b[i]) > tol) {
			if (mismatches < 5) {
				printf("  mismatch at %zu: gpu=%f cpu=%f\n", i, a[i], b[i]);
			}
			mismatches++;
		}
	}
	return mismatches;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(void)
{
	// check and set device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	const int isize      = ISIZE;
	const int ksize      = KSIZE;
	const int osize      = OSIZE;
	const int width      = WIDTH;
	const int mask_width = MASK_WIDTH;

	const int threads = THREADS_PER_BLOCK;
	const int blocks  = (osize + threads - 1) / threads;   // cover every output

	printf("size of i array: %d\n", isize);
	printf("size of k array: %d\n", ksize);
	printf("size of o array: %d\n", osize);
	printf("mask width: %d (radius %d)\n", mask_width, MASK_RADIUS);
	printf("threads per block: %d\n", threads);
	printf("number of blocks: %d\n", blocks);
	printf("shared tile per block: %d floats (%zu bytes)\n",
	       TILE_SIZE, (size_t)TILE_SIZE * sizeof(float));

	// initialize arrays
	vector<float> host_i(isize);
	vector<float> host_k(ksize);
	vector<float> host_o(osize);
	vector<float> cpu_o(osize);

	srand((int)time(0));
	generate(host_i.begin(), host_i.end(), []() { return (float)(rand() % 9); });
	generate(host_k.begin(), host_k.end(), []() { return (float)(rand() % 9); });

	// device memory
	float *dev_i = nullptr, *dev_o = nullptr;
	CHECK(cudaMalloc(&dev_i, isize * sizeof(float)));
	CHECK(cudaMalloc(&dev_o, osize * sizeof(float)));

	CHECK(cudaMemcpy(dev_i, host_i.data(), isize * sizeof(float), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(K, host_k.data(), mask_width * sizeof(float)));

	dim3 block(threads);
	dim3 grid(blocks);

	// Warm-up launch.  The first kernel launch in a process pays for
	// context creation and module loading, which would otherwise be
	// counted as compute time.
	convolution_1D_tiled_kernel<<<grid, block>>>(dev_i, dev_o, width, osize);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	// Timed GPU run
	cudaEvent_t start, kernel_done, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&kernel_done));
	CHECK(cudaEventCreate(&stop));

	CHECK(cudaEventRecord(start, 0));
	convolution_1D_tiled_kernel<<<grid, block>>>(dev_i, dev_o, width, osize);
	CHECK(cudaGetLastError());
	CHECK(cudaEventRecord(kernel_done, 0));
	CHECK(cudaMemcpy(host_o.data(), dev_o, osize * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaEventRecord(stop, 0));
	CHECK(cudaEventSynchronize(stop));

	float kernelTime = 0.0f, GPUtime = 0.0f;
	CHECK(cudaEventElapsedTime(&kernelTime, start, kernel_done));
	CHECK(cudaEventElapsedTime(&GPUtime, start, stop));

	// Timed CPU run (single thread)
	auto cpu_start = std::chrono::steady_clock::now();
	convolution_1D_CPU(host_i, host_k, cpu_o, mask_width, width);
	auto cpu_stop = std::chrono::steady_clock::now();
	float CPUtime = std::chrono::duration<float, std::milli>(cpu_stop - cpu_start).count();

	printf("Kernel time on GPU: %3.6f ms \n", kernelTime);
	printf("Compute time on GPU: %3.6f ms  (kernel + device-to-host copy)\n", GPUtime);
	printf("Compute time on CPU: %3.6f ms \n", CPUtime);

	// Check result element by element
	int mismatches = countMismatches(host_o, cpu_o);
	if (mismatches == 0)
		printf("Check Result: Arrays matched\n");
	else
		printf("Check Result: Arrays do not match (%d of %d elements differ)\n", mismatches, osize);

	// Free memory
	CHECK(cudaEventDestroy(start));
	CHECK(cudaEventDestroy(kernel_done));
	CHECK(cudaEventDestroy(stop));
	CHECK(cudaFree(dev_i));
	CHECK(cudaFree(dev_o));

	return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
