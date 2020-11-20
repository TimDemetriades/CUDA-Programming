// CUDA List Scan/Prefix Sum
// Tim Demetriades
// CPE 810 - GPU & Multicore Programming
// Professor Feng
// Stevens Institute of Technology

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"

//for __syncthreads()
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

#include <stdio.h>
#include <cstdlib>
#include <time.h>	// for CPU timer
#include <math.h>	// for power function

// Thread block size
#define BLOCK_SIZE 2048 / 2

// Section size
#define SECTION_SIZE 2048

// Number of iterations for GPU and CPU List Scans
#define NITER 100

// Kernel function for List Scan (part 1)
__global__ void ListScan_GPU_1(unsigned int* device_input, unsigned int* device_output, unsigned int input_size, unsigned int* S) {

	// Shared memory
	__device__ __shared__ int device_output_shared[SECTION_SIZE];
	int tid = 2 * blockIdx.x * blockDim.x + threadIdx.x;			// Set thread index
	if (tid < input_size) {
		device_output_shared[threadIdx.x] = device_input[tid];		// Move values from global to shared memory
	}
	if (tid + blockDim.x < input_size) {
		device_output_shared[threadIdx.x + blockDim.x] = device_input[tid + blockDim.x];
	}

	// Reduction phase
	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < SECTION_SIZE) {
			device_output_shared[index] += device_output_shared[index - stride];
		}
	}

	// Post reduction reverse phase
	for (unsigned int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < SECTION_SIZE) {
			device_output_shared[index + stride] += device_output_shared[index];
		}
	}

	// Move output values from shared memory to global memory
	__syncthreads();
	if (tid < input_size) {
		device_output[tid] = device_output_shared[threadIdx.x];
	}
	if (tid + blockDim.x < input_size) {
		device_output[tid + blockDim.x] = device_output_shared[threadIdx.x + blockDim.x];
	}

	// Fill S array with last value of each section
	__syncthreads();
	if (threadIdx.x == blockDim.x - 1) {
		S[blockIdx.x] = device_output_shared[SECTION_SIZE - 1];
	}
}

// Kernel function for List Scan (part 2)
__global__ void ListScan_GPU_2(unsigned int* device_S, unsigned int input_size) {
	
	// Shared memory
	__device__ __shared__ int device_output_shared[SECTION_SIZE];
	int tid = 2 * blockIdx.x * blockDim.x + threadIdx.x;			// Set thread index
	if (tid < input_size) {
		device_output_shared[threadIdx.x] = device_S[tid];			// Move values from global to shared memory
	}
	if (tid + blockDim.x < input_size) {
		device_output_shared[threadIdx.x + blockDim.x] = device_S[tid + blockDim.x];
	}

	// Reduction phase
	for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < SECTION_SIZE) {
			device_output_shared[index] += device_output_shared[index - stride];
		}
	}

	// Post reduction reverse phase
	for (unsigned int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
		__syncthreads();
		int index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < SECTION_SIZE) {
			device_output_shared[index + stride] += device_output_shared[index];
		}
	}

	// Move output values from shared memory to global memory
	__syncthreads();
	if (tid < input_size) {
		device_S[tid] = device_output_shared[threadIdx.x];
	}
	if (tid + blockDim.x < input_size) {
		device_S[tid + blockDim.x] = device_output_shared[threadIdx.x + blockDim.x];
	}
}

// Kernel function for List Scan (part 3)
__global__ void ListScan_GPU_3(unsigned int* device_output, unsigned int* device_S) {

	int tid = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
	device_output[tid] += device_S[blockIdx.x - 1];
}

// CPU Sequential List Scan (for comparing with GPU List Scan and veryfing results)
void ListScan_CPU(unsigned int* host_input, unsigned int* host_output_cpu, unsigned int input_size) {
	int accumulator = host_input[0];		// Set accumulator to first value of input
	host_output_cpu[0] = accumulator;			// Set first value of output to accumulator
	for (int i = 1; i < input_size; i++) {
		accumulator += host_input[i];		// Accumulator = accumulator + current input(next value)
		host_output_cpu[i] = accumulator;		// Current output = accumulator
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
	unsigned int input_size = 0;

	if (argc == 3) {		// 3 arguments expected (filename, -i, <dim>
		if (atoi(argv[2]) <= 0 || atoi(argv[2]) > 2048 * 65535) {
			printf("\nPlease make sure <dim> is between 1 and 2048 * 65535.\n");
			exit(EXIT_FAILURE);
		}
		// Set input size
		input_size = atoi(argv[2]);
		printf("\nThe input is %d integers long\n", input_size);
	}
	else if (argc > 3) {
		printf("\nToo many arguments provided.\nEnter arguments like this: \n");
		printf("-i <dim> \n");
		exit(EXIT_FAILURE);
	}
	else {
		printf("\n2 arguments expected.\nEnter arguments like this: \n");
		printf("-i <dim> \n");
		exit(EXIT_FAILURE);
	}

	// Size in bytes of input
	size_t input_bytes = input_size * sizeof(int); // unsigned int = int = 4 bytes

	// Size in bytes of S (auxiliary array that holds reduction of each scan block)
	size_t S_bytes = input_size / SECTION_SIZE * sizeof(int);

	// Allocate host memory for input and output
	unsigned int* host_input;
	unsigned int* host_output;
	unsigned int* host_S;
	unsigned int* host_output_cpu;
	host_input = (unsigned int*)malloc(input_bytes);
	host_output = (unsigned int*)malloc(input_bytes);	
	host_S = (unsigned int*)malloc(S_bytes);
	host_output_cpu = (unsigned int*)malloc(input_bytes);

	// Allocate device memory for input and output
	unsigned int* device_input;
	unsigned int* device_output;
	unsigned int* device_S;
	checkCudaErrors(cudaMalloc((void**)&device_input, input_bytes));
	checkCudaErrors(cudaMalloc((void**)&device_output, input_bytes));
	checkCudaErrors(cudaMalloc((void**)&device_S, S_bytes));

	// Initialize input with random ints between 0~1024
	srand((unsigned int)time(NULL));		// Assigns seed to make random numbers change
	for (int i = 0; i < input_size; i++) {
		host_input[i] = rand() % 1024;		// Not including 1024
		//host_input[i] = 1;		// for testing
	}

	//// Print input
	//printf("\nInput:\n");
	//for (int i = 0; i < input_size; i++) {
	//	printf("\nValue[%d] = %u", i, host_input[i]);
	//	if (i == input_size - 1) {
	//		printf("\n");
	//	}
	//}

	// Copy input values from host to device
	checkCudaErrors(cudaMemcpy(device_input, host_input, input_bytes, cudaMemcpyHostToDevice));		//dest, source, size in bytes, direction of transfer

	// Set grid and thread block sizes
	int block_size = BLOCK_SIZE;								// Threads per block
	int grid_size = ceil(input_size / SECTION_SIZE) + 1;		// Blocks per grid
	dim3 dim_block(block_size);
	dim3 dim_grid(grid_size);

	// Record the start event (for timing GPU calculations)
	cudaStream_t stream;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	checkCudaErrors(cudaStreamSynchronize(stream));
	checkCudaErrors(cudaEventRecord(start, stream));

	int nIter = NITER;	// How many times to run kernel

	printf("\nStarting List Scan on GPU\n");

	// Launch kernel (repeat nIter times so we can obtain average run time)
	for (int i = 0; i < nIter; i++) {
		ListScan_GPU_1 << <dim_grid, dim_block >> > (device_input, device_output, input_size, device_S);
		//ListScan_GPU_2 << <dim_grid, dim_block >> > (device_S, input_size);
		//ListScan_GPU_3 << <dim_grid, dim_block >> > (device_output, device_S);
	}

	printf("\n\GPU List Scan Complete\n");

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	float msecPerHistogram = msecTotal / nIter;
	printf("\nGPU Histogram Computation took %.3f msec\n", msecPerHistogram);
	printf("\nThreads per block = %d, Blocks per grid = %d, Total threads = %d\n", block_size, grid_size, block_size * grid_size);

	// Copy output values from device to host
	checkCudaErrors(cudaMemcpy(host_output, device_output, input_bytes, cudaMemcpyDeviceToHost));	//dest, source, size in bytes, direction of transfer

	// Copy S values from device to host
	checkCudaErrors(cudaMemcpy(host_S, device_S, S_bytes, cudaMemcpyDeviceToHost));	//dest, source, size in bytes, direction of transfer

	// Make sure GPU results are correct depending on number of iterations
	for (int i = 0; i < input_size; i++) {
		host_output[i] / nIter;
		host_S[i] / nIter;
	}

	//// Print GPU results
	//printf("\nGPU Results:\n");
	//for (int i = 0; i < input_size; i++) {
	//	printf("\nValue[%d] = %u", i, host_output[i]);
	//	if (i == input_size - 1) {
	//		printf("\n");
	//	}
	//}

	//// Print S values
	//printf("\nS Values:\n");
	//for (int i = 0; i < input_size / SECTION_SIZE; i++) {
	//	printf("\nValue[%d] = %u", i, host_S[i]);
	//	if (i == input_size / SECTION_SIZE - 1) {
	//		printf("\n");
	//	}
	//}

	//Start CPU timer
	double time_taken_cpu = 0.0;
	clock_t begin_cpu = clock();

	// Calculate list scan on CPU
	printf("\nStarting List Scan on CPU\n");
	for (int i = 0; i < nIter; i++) {		// Repeat CPU computation same amount of times as GPU computation
		ListScan_CPU(host_input, host_output_cpu, input_size);
	}
	printf("\nCPU List Scan Complete\n");

	clock_t end_cpu = clock();
	time_taken_cpu += ((double)(end_cpu - begin_cpu) / CLOCKS_PER_SEC * 1000) / nIter;	// in milliseconds
	printf("\nCPU List Scan took %.3f msec\n", time_taken_cpu);

	// Make sure CPU results are correct depending on number of iterations
	for (int i = 0; i < input_size; i++) {
		host_output_cpu[i] / nIter;
	}

	//// Print CPU results
	//printf("\nCPU Results:\n");
	//for (int i = 0; i < input_size; i++) {
	//	printf("\nValue[%d] = %u", i, host_output_cpu[i]);
	//	if (i == input_size - 1) {
	//		printf("\n");
	//	}
	//}

	// Check if GPU and CPU results match
	bool check = 0;
	for (int i = 0; i < input_size; i++) {				// For every value in the outputs
		if (host_output[i] != host_output_cpu[i]) {		// Check if they match and if not set a flag
			check = 1;
		}
	}
	if (check == 1) {
		printf("\nGPU and CPU results do not match!\n");
	}
	else {
		printf("\nGPU and CPU results match!\n");
	}

	// Free memory in device
	cudaFree(device_input);
	cudaFree(device_output);
	cudaFree(device_S);

	// Free memory in host
	free(host_input);
	free(host_output);
	free(host_S);
	free(host_output_cpu);
}