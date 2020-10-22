// CUDA Histogram Computation
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
#include <time.h>

// Statically allocate shared memory
//#define SHARED_MEM_SIZE 16 * 16	// Dimensions of each block/tile

// Kernel function for Histogram Computation
__global__ void Histogram_Computation(unsigned int* device_input, unsigned int* device_bins, unsigned int input_size, unsigned int num_bins) {

	printf("Test");

	// Get thread id
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	int DIV = (input_size + num_bins - 1) / num_bins;

	//printf("tid = %d\n", tid);
	//printf("input_size = %d\n", input_size);

	// Boundary condition
	if (tid < input_size) {
		int bin = device_input[tid] / DIV;
		atomicAdd(&device_bins[bin], 1);
		printf("Test");
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
	if (argc == 4) {		// 4 arguments expected (filename, -i, <BinNum>, <VecDim>)
		printf("\n");
		printf("There will be %s bins \n", argv[2]);
		printf("Input vector is %s elements long \n", argv[3]);
	}
	else if (argc > 4) {
		printf("Too many arguments provided. \n");
		printf("Enter arguments like this: \n");
		printf("-i <BinNum> <VecDim> \n");
		exit(EXIT_FAILURE);
	}
	else {
		printf("3 arguments expected. \n");
		printf("Enter arguments like this: \n");
		printf("-i <BinNum> <VecDim> \n");
		exit(EXIT_FAILURE);
	}

	printf("\nStarting Histogram Computation on GPU\n");

	// Set number of bins
	int num_bins_string;
	num_bins_string = atoi(argv[2]);
	int num_bins = num_bins_string;
	//int num_bins = atoi(argv[2]);

	// Set number of input elements
	int input_size_string;
	input_size_string = atoi(argv[3]);
	int input_size = input_size_string;
	//int input_size = atoi(argv[3]);

	// Set number of elements per bin
	int bin_size = ceil(input_size / num_bins);		// Padded in case num of bins does not equally divide num of input elements
	//(input_size + num_bins - 1) / num_bins   ->   alternate method

	// Size in bytes of input vector
	size_t input_bytes = input_size * sizeof(int);

	// Size in bytes of bins
	size_t bin_bytes = num_bins * sizeof(int);

	// Allocate host memory for input vector and bins
	unsigned int* host_input;
	unsigned int* host_bins;
	host_input = (unsigned int*)malloc(input_bytes);
	host_bins = (unsigned int*)malloc(bin_bytes);

	// Allocate device memory for input vector and bins
	unsigned int* device_input;
	unsigned int* device_bins;
	checkCudaErrors(cudaMalloc((void**)&device_input, input_bytes));
	checkCudaErrors(cudaMalloc((void**)&device_bins, bin_bytes));

	// Initialize input vector with ints between 0~1024
	for (int i = 0; i < input_size; i++) {
		host_input[i] = rand() % 1025;
	}

	// Initialize bins with 0s
	for (int i = 0; i < num_bins; i++) {
		host_bins[i] = 0;
	}

	// Print input vector
	printf("\nInput vector:\n");
	for (int i = 0; i < input_size; i++) {
		printf("%d\t", host_input[i]);
	}

	// Print input vector
	printf("\nBins:\n");
	for (int i = 0; i < num_bins; i++) {
		printf("%d\t", host_bins[i]);
	}

	// Copy matrix values from host to device
	checkCudaErrors(cudaMemcpy(device_input, host_input, input_bytes, cudaMemcpyHostToDevice));		// dest, source, size in bytes, direction of transfer
	checkCudaErrors(cudaMemcpy(device_bins, host_bins, bin_bytes, cudaMemcpyHostToDevice));		// dest, source, size in bytes, direction of transfer

	//// Print input vector
	//printf("\nInput vector - Device:\n");
	//for (int i = 0; i < input_size; i++) {
	//	printf("%d\t", device_bins[i]);
	//}

	// Set Grid and Block sizes
	int block_size = 512;		// Threads per block
	int grid_size = ceil(input_size / block_size) + 1;
	dim3 dim_block(block_size, block_size);
	dim3 dim_grid(grid_size, grid_size);

	// Record the start event (for timing GPU calculations)
	cudaStream_t stream;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	checkCudaErrors(cudaStreamSynchronize(stream));
	checkCudaErrors(cudaEventRecord(start, stream));

	int nIter = 100;	// How many times to run kernel

	// Launch kernel (repeat nIter times so we can obtain average run time)
	for (int i = 0; i < nIter; i++) {
		Histogram_Computation << < dim_grid, dim_block >> > (device_input, device_bins, input_size, num_bins);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	printf("\nGPU Histogram Computation Complete\n");

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	float msecPerHistogram = msecTotal / nIter;
	printf("\nTime = %.3f msec", msecPerHistogram);

	// Copy matrix values from device to host
	checkCudaErrors(cudaMemcpy(host_bins, device_bins, bin_bytes, cudaMemcpyDeviceToHost));

	for (int i = 0; i < num_bins; i++) {
		printf("\nBin Output = %d", host_bins[i]);
	}

	int tmp = 0;
	for (int i = 0; i < num_bins; i++) {
		tmp += host_bins[i];
	}

	printf("\nOutput = %d", tmp);

	// Free memory in device
	checkCudaErrors(cudaFree(device_input));
	checkCudaErrors(cudaFree(device_bins));

	// Free memory in host
	free(host_input);
	free(host_bins);
}
