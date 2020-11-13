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
#define BLOCK_SIZE 256

// Statically allocate shared memory
#define SHARED_MEM_SIZE 256 * 4		// Size of each thread block * Size of int

// Kernel function for List Scan
__global__ void ListScan_GPU(unsigned int* device_input, unsigned int* device_output, unsigned int input_size) {

}

void ListScan_CPU(unsigned int* host_input, unsigned int* host_output, unsigned int input_size) {

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
		printf("\nThe input is %d integers long.\n", input_size);
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

	// Allocate host memory for input and output
	unsigned int* host_input;
	unsigned int host_output;
	unsigned int* host_output_cpu;
	host_input = (unsigned int*)malloc(input_bytes);
	host_output = (unsigned int)malloc(sizeof(int));		// output is just a single int
	host_output_cpu = (unsigned int*)malloc(sizeof(int));

	// Allocate device memory for input and output
	unsigned int* device_input;
	unsigned int device_output;
	checkCudaErrors(cudaMalloc((void**)&device_input, input_bytes));
	checkCudaErrors(cudaMalloc(&device_output, sizeof(int)));

	// Initialize input with random ints between 0~1024
	srand((unsigned int)time(NULL));		// Assigns seed to make random numbers change
	for (int i = 0; i < input_size; i++) {
		//host_input[i] = rand() % 1024;		// Not including 1024
		host_input[i] = 1;		// for testing
	}

	// Print input
	for (int i = 0; i < input_size; i++) {
		printf("%d\t", host_input[i]);
	}

	// Copy input values from host to device
	checkCudaErrors(cudaMemcpy(device_input, host_input, input_bytes, cudaMemcpyHostToDevice));		//dest, source, size in bytes, direction of transfer

	// Set grid and thread block sizes
	int block_size = BLOCK_SIZE;						// Threads per block
	int grid_size = ceil(input_size / block_size);		// Blocks per grid

	// Record the start event (for timing GPU calculations)
	cudaStream_t stream;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	checkCudaErrors(cudaStreamSynchronize(stream));
	checkCudaErrors(cudaEventRecord(start, stream));

	int nIter = 1;	// How many times to run kernel

	printf("\nStarting List Scan on GPU\n");

	// Launch kernel (repeat nIter times so we can obtain average run time)
	for (int i = 0; i < nIter; i++) {
		ListScan_GPU << <dim_grid, dim_block >> > (device_input, device_output, input_size);
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

	// Copy matrix values from device to host
	checkCudaErrors(cudaMemcpy(host_output, device_output, sizeof(int), cudaMemcpyDeviceToHost));

	// Print GPU results
	printf("\nGPU Results:\n");
	printf("\nSum = %d", host_input[0]);

}