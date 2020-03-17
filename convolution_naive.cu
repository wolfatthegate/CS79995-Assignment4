/**
* This program computes the 1-D convolution with 
* naive algorithm written in parallel. 
*/
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <algorithm>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

#define ISIZE 2000
#define KSIZE 500
#define BLOCK_SIZE 10

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	printf("Error at %s:%d\n", __FILE__, __LINE__);\
	return EXIT_FAILURE;}} while(0)

#define CHECK(x) do { if((x) !=cudaSuccess) { \
	printf("Error at %s:%d\n", __FILE__, __LINE__); \
	return EXIT_FAILURE;}} while(0)

// kernel function 
__global__ void convolution_1D_basic_kernal(float *I, float*K, float *O, 
												int Mask_Width, int Width){
	
	int i = blockIdx.x * blockDim.x + threadIdx.x; 

	float Ovalue = 0; 
	int N_start_point = i - (Mask_Width/2); 
	for (int j = 0; j < Mask_Width; j++) {
		if (N_start_point + j >=0 && N_start_point + j < Width) {
			Ovalue += I[N_start_point + j] * K[j];
		}
	}
	O[i] = Ovalue; 
} 

// cpu function
void convolution_1D_basic_kernal_CPU(vector<float> &host_i, vector<float> &host_k, 
											int &cpuRef, int Mask_Width, int Width, int size){
	cpuRef = 0.0;
	for (int i = 0; i < size; i++){
		int N_start_point = i - (Mask_Width/2); 
		for (int j = 0; j < Mask_Width; j++) {
			if (N_start_point + j >=0 && N_start_point + j < Width) {
				cpuRef += host_i[N_start_point + j] * host_k[j];
			}
		}
	}
}

// function for checking gpu reference array
void checkArray(vector<float> &host_o, int &gpuRef,int size){

	gpuRef = 0; 
	for (int x = 0; x < size ; x ++) {
		gpuRef += host_o[x];
	}
}

// main function
int main(void){

	// check and set device
	int dev = 0; 
	cudaDeviceProp deviceProp; 
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev)); 

	int cpuRef = 0; 
	int gpuRef = 0; 

	int isize = ISIZE; // size of function f or array f
	int ksize = KSIZE; // size of function g or array g
	int osize = ISIZE + KSIZE - 1;  // size of output function f*g. 
	int blockSize = BLOCK_SIZE; 

	int width = (int) ISIZE/10;
	int mask_width = 2*width +1;  

	printf("size of i array: %d\n", isize); 
	printf("size of k array: %d\n", ksize); 
	printf("size of block: %d\n", BLOCK_SIZE);

	// initialize array
	vector<float> host_i (isize); 
	vector<float> host_k (ksize); 
	vector<float> host_o (osize);
	vector<float> cpuRefArr (osize);
	
	// initialize random number
	srand ((int)time(0));

	// generate elements in arrays
	generate(host_i.begin(), host_i.end(), []() { return rand() % 9; });
	generate(host_k.begin(), host_k.end(), []() { return rand() % 9; });

	//memory allocation

	float *dev_i, *dev_k, *dev_o; 
	cudaMalloc(&dev_i, isize * sizeof(float));
	cudaMalloc(&dev_k, ksize * sizeof(float));
	cudaMalloc(&dev_o, osize * sizeof(float));

	//cudaMemcopyHostToDevice
	cudaMemcpy(dev_i, host_i.data(), isize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_k, host_k.data(), ksize * sizeof(float), cudaMemcpyHostToDevice);

	//initalize dimension
	dim3 block(isize/blockSize);
	dim3 grid(blockSize);

	float GPUtime, CPUtime; 
	cudaEvent_t start, stop; 

	// timer starts for GPU calculation
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0); 

	//kernel launch
	convolution_1D_basic_kernal <<< grid, block >>> (dev_i, dev_k, dev_o, mask_width, width); 

	//cudaMemcopyDeviceToHost
	cudaMemcpy(host_o.data(), dev_o, osize * sizeof(float), cudaMemcpyDeviceToHost); 

	// timer stops for GPU calculation
	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&GPUtime, start, stop); 

	// timer starts for CPU calculation
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 
	cudaEventRecord(start, 0); 

	//calculate on CPU and check the result (single thread)
	convolution_1D_basic_kernal_CPU(host_i, host_k, cpuRef, mask_width, width, osize);

	cudaEventRecord(stop, 0); 
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&CPUtime, start, stop); 

	printf("Compute time on GPU: %3.6f ms \n", GPUtime); 
	printf("Compute time on CPU: %3.6f ms \n", CPUtime); 

	//checkResult
	checkArray(host_o, gpuRef, osize);

	double epsilon = 1.0E-8; 
	if(abs(cpuRef - gpuRef)<epsilon)
		printf("Check Result: Arrays matched\n"); 
	else 
		printf("Check Result: Arrays do not match\n"); 

	//Free Memory
	cudaFree(dev_i);
	cudaFree(dev_k);
	cudaFree(dev_o);

	return(0);
}