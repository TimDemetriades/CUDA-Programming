// CUDA Histogram Computation
// Tim Demetriades
// CPE 810 - GPU & Multicore Programming
// Professor Feng
// Stevens Institute of Technology

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"

//for __syncthreads() and atomicadd
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

#include <stdio.h>
#include <cstdlib>
#include <time.h>	// for CPU timer
#include <math.h>	// for power function

// Statically allocate shared memory
#define SHARED_MEM_SIZE 16	// Size of each block

// Kernel function for Histogram Computation using interleaved portioning (memory coalescing)
__global__ void Histogram_GPU_1(unsigned int* device_input, unsigned int* device_bins, unsigned long input_size, unsigned int bin_size) {

	// Get thread id
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for (unsigned int i = tid; i < input_size; i += (blockDim.x * gridDim.x)) {	// blockDim.x * gridDim.x = total number of threads for each kernel invocation
		if (device_input[i] >= 0 && device_input[i] < 1024) {	// Boundary condition
			atomicAdd(&device_bins[device_input[i] / bin_size], 1);
		}
	}
}

// Kernel function for Histogram Computation using shared memory (privatization)
__global__ void Histogram_GPU_2(unsigned int* device_input, unsigned int* device_bins, unsigned long input_size, unsigned int bin_size, unsigned int num_bins) {

	// Get thread id
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Private bins
	__device__ __shared__ int device_bins_private[SHARED_MEM_SIZE];
	// Initialize private bins to 0
	for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
		device_bins_private[binIdx] = 0;
	}
	__syncthreads();

	// Compute histogram
	for (unsigned int i = tid; i < input_size; i += blockDim.x * gridDim.x) {
		if (device_input[i] >= 0 && device_input[i] < 1024) {	// Boundary condition
			atomicAdd(&device_bins_private[device_input[i] / bin_size], 1);
		}
	}
	__syncthreads();

	// Move from shared to global memory
	for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
		atomicAdd(&device_bins[binIdx], device_bins_private[binIdx]);
	}
}

// Kernel function for Histogram Computation using shared memory (privatization) and aggregation
__global__ void Histogram_GPU_3(unsigned int* device_input, unsigned int* device_bins, unsigned long input_size, unsigned int bin_size, unsigned int num_bins) {

	// Get thread id
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Private bins
	__device__ __shared__ int device_bins_private[SHARED_MEM_SIZE];
	// Initialize private bins to 0
	for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
		device_bins_private[binIdx] = 0;
	}
	__syncthreads();

	unsigned int prev_index = -1;		// Tracks index of histogram element whose updates have been aggregated (-1 so it won't have chance of matching value in bin)
	unsigned int accumulator = 0;		// Keeps track of number of updates aggregated so far (0 means no updates have been aggregated)

	// Compute histogram
	for (unsigned int i = tid; i < input_size; i += blockDim.x * gridDim.x) {
		if (device_input[i] >= 0 && device_input[i] < 1024) {	// Boundary condition
			unsigned int current_index = device_input[i] / bin_size;		
			if (current_index != prev_index) {		// Compare index of histogram element to be updated with index of one currently being aggregated
				if (accumulator >= 0) {
					atomicAdd(&device_bins_private[device_input[i] / bin_size], accumulator);		// Current and previous are different, so add accumulator to value in bin
					accumulator = 1;
					prev_index = current_index;
				}
			}
			else {
				accumulator ++;		// Current and previous match so increment accumulator
			}
		}
	}
	__syncthreads();

	// Move from shared to global memory
	for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
		atomicAdd(&device_bins[binIdx], device_bins_private[binIdx]);
	}
}

// Histogram Computation on CPU
void Histogram_CPU(unsigned int* host_input, unsigned int input_size, unsigned int bin_size, unsigned int * host_bins) {
	for (int i = 0; i < input_size; i++) {
		host_bins[host_input[i] / bin_size]++;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
	unsigned int num_bins = 0;
	unsigned long input_size = 0;

	if (argc == 4) {		// 4 arguments expected (filename, -i, <BinNum>, <VecDim>)
		int bin_exponent = atoi(argv[2]);
		if (bin_exponent < 2 || bin_exponent > 8) {
			printf("\nPlease make sure <BinNum> is between 2 and 8.\n");
			exit(EXIT_FAILURE);
		}
		// Set number of bins 
		num_bins = pow(2, bin_exponent);
		printf("\nThere will be %d bins \n", num_bins);

		// Set number of input elements
		input_size = atoi(argv[3]);
		printf("Input vector is %d elements long \n", input_size);
	}
	else if (argc > 4) {
		printf("\nToo many arguments provided. \n");
		printf("Enter arguments like this: \n");
		printf("-i <BinNum> <VecDim> \n");
		exit(EXIT_FAILURE);
	}
	else {
		printf("\n3 arguments expected. \n");
		printf("Enter arguments like this: \n");
		printf("-i <BinNum> <VecDim> \n");
		exit(EXIT_FAILURE);
	}

	// Set number of elements per bin
	unsigned int bin_size = 1024 / num_bins;		// 1024 is the max possible input element
	printf("Each bin will contain %u elements\n", bin_size);

	// Size in bytes of input vector
	size_t input_bytes = input_size * sizeof(int);

	// Size in bytes of bins
	size_t bin_bytes = num_bins * sizeof(int);

	// Allocate host memory for input vector and bins
	unsigned int* host_input;
	unsigned int* host_bins;
	unsigned int* host_bins_cpu;
	host_input = (unsigned int*)malloc(input_bytes);
	host_bins = (unsigned int*)malloc(bin_bytes);
	host_bins_cpu = (unsigned int*)malloc(bin_bytes);

	// Allocate device memory for input vector and bins
	unsigned int* device_input;
	unsigned int* device_bins;
	checkCudaErrors(cudaMalloc((void**)&device_input, input_bytes));
	checkCudaErrors(cudaMalloc((void**)&device_bins, bin_bytes));

	// Initialize input vector with ints between 0~1024
	srand((unsigned int)time(NULL));		// Assigns seed to make random numbers change
	for (int i = 0; i < input_size; i++) {
		host_input[i] = rand() % 1024;
	}

	// Initialize bins with 0s
	for (int i = 0; i < num_bins; i++) {
		host_bins[i] = 0;
	}
	for (int i = 0; i < num_bins; i++) {
		host_bins_cpu[i] = 0;
	}

	// Print input vector
	/*printf("\nInput vector:\n");
	for (int i = 0; i < input_size; i++) {
		printf("%d\t", host_input[i]);
	}*/

	// Copy matrix values from host to device
	checkCudaErrors(cudaMemcpy(device_input, host_input, input_bytes, cudaMemcpyHostToDevice));		// dest, source, size in bytes, direction of transfer

	// Set Grid and Block sizes
	int block_size = 16;		// Threads per block
	int grid_size = input_size / block_size;
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

	int nIter = 1;	// How many times to run kernel

	printf("\nStarting Histogram Computation on GPU\n");

	// Launch kernel (repeat nIter times so we can obtain average run time)
	for (int i = 0; i < nIter; i++) {
		//Histogram_GPU_1<<<dim_grid, dim_block>>>(device_input, device_bins, input_size, bin_size);
		Histogram_GPU_2<<<dim_grid, dim_block>>>(device_input, device_bins, input_size, bin_size, num_bins);
		//Histogram_GPU_3<<<dim_grid, dim_block>>>(device_input, device_bins, input_size, bin_size, num_bins);
	}

	printf("\n\GPU Histogram Computation Complete\n");

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	float msecPerHistogram = msecTotal / nIter;
	printf("\nGPU Histogram Computation took %.3f msec\n", msecPerHistogram);

	// Copy matrix values from device to host
	checkCudaErrors(cudaMemcpy(host_bins, device_bins, bin_bytes, cudaMemcpyDeviceToHost));

	//// Print GPU results
	//printf("\nGPU Results: \n");
	//for (int i = 0; i < num_bins; i++) {
	//	printf("\nBins %d = %u", i, host_bins[i]);
	//}

	//int sum_bins = 0;
	//for (int i = 0; i < num_bins; i++) {
	//	sum_bins += host_bins[i];
	//}
	//printf("\n\nSummation of all the bins = %d\n", sum_bins);

	//Start CPU timer
	double time_taken_cpu = 0.0;
	clock_t begin_cpu = clock();

	// Calculate histogram on CPU
	printf("\nStarting Histogram Computation on CPU\n");
	Histogram_CPU(host_input, input_size, bin_size, host_bins_cpu);
	printf("\nCPU Histogram Computation Complete\n");

	clock_t end_cpu = clock();
	time_taken_cpu += (double)(end_cpu - begin_cpu) / CLOCKS_PER_SEC * 1000;	// in milliseconds
	printf("\nCPU Histogram Computation took %.3f msec\n", time_taken_cpu);

	//// Print CPU results
	//printf("\nCPU Results: \n");
	//for (int i = 0; i < num_bins; i++) {
	//	printf("\nBins %d = %u", i, host_bins_cpu[i]);
	//}

	//int sum_bins_cpu = 0;
	//for (int i = 0; i < num_bins; i++) {
	//	sum_bins_cpu += host_bins_cpu[i];
	//}
	//printf("\n\nSummation of all the bins = %d\n", sum_bins_cpu);

	// Check if GPU and CPU histograms match
	bool check = 0;
	for (int i = 0; i < num_bins; i++) {			// For every value in the arrays
		if (host_bins[i] != host_bins_cpu[i]) {		// Check if they match and if not set a flag
			check = 1;
		}
	}

	if (check == 1) {
		printf("\nGPU and CPU histograms do not match!\n");
	}
	else {
		printf("\nGPU and CPU histograms match!\n");
	}

	// Free memory in device
	cudaFree(device_input);
	cudaFree(device_bins);

	// Free memory in host
	free(host_input);
	free(host_bins);
}
