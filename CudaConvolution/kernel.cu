// CUDA Convolution
// Tim Demetriades
// CPE 810 - GPU & Multicore Programming
// Professor Feng
// Stevens Institute of Technology

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"

////for __syncthreads()
//#ifndef __CUDACC__ 
//#define __CUDACC__
//#endif
//#include <device_functions.h>

#include <stdio.h>
#include <cstdlib>
#include <time.h>	// for CPU timer

#define MAX_MASK_SIZE 128							// Set max mask size (arbitrary)
__constant__ int device_mask[MAX_MASK_SIZE];		// Initialize mask as constant memory

// Kernel function for Histogram Computation using interleaved portioning (memory coalescing)
__global__ void Convolution_GPU(unsigned int* device_input, unsigned int* device_output, unsigned int input_x, unsigned int input_y, unsigned int mask_size) {

	// Calculate row and column indexes for each thread (location of the thread)
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	//int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	//printf("tid = %d\n", tid);

	// Start point for rows and columns
	int row_start = tid - (mask_size / 2);
	//int col_start = col - (mask_size / 2);

	// Temp value for result accumulation
	int temp = 0;

	// Convolution
	for (int i = 0; i < mask_size; i++) {
		if (((row_start + i) >= 0) && (row_start + i < mask_size)) {
			temp += device_input[row_start + i] * device_mask[i];
		}
	}

	device_output[tid] = temp;
}

// Histogram Computation on CPU
void Histogram_CPU(unsigned int* host_input, unsigned int input_size, unsigned int bin_size, unsigned int* host_bins) {
	for (int i = 0; i < input_size; i++) {
		host_bins[host_input[i] / bin_size]++;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
	unsigned int input_x = 0;
	unsigned int input_y = 0;
	unsigned int mask_size = 0;

	if (argc == 7) {		// 7 arguments expected (filename, -i, <dimX>, -j, <dimY>, -k, <dimK>)
		// Get input x dimension from user input
		input_x = atoi(argv[2]);
		if (input_x < 1) {
			printf("\nPlease make sure <dimX> is an integer greater than 0.\n");
			exit(EXIT_FAILURE);
		}

		// Get input y dimension from user input
		input_y = atoi(argv[4]);
		if (input_y < 1) {
			printf("\nPlease make sure <dimY> is an integer greater than 0.\n");
			exit(EXIT_FAILURE);
		}
		// Get mask size from user input
		mask_size = atoi(argv[6]);
		if (mask_size < 1) {
			printf("\nPlease make sure <dimK> is an integer greater than 0.\n");
			exit(EXIT_FAILURE);
		}
		printf("\nInput matrix is %u by %u \n", input_x, input_y);
		printf("Mask size is %u \n", mask_size);
	}
	else if (argc > 7) {
		printf("\nToo many arguments provided. \n");
		printf("Enter arguments like this: \n");
		printf("-i <dimX> -j <dimY> -k <dimK> \n");
		exit(EXIT_FAILURE);
	}
	else {
		printf("\n6 arguments expected. \n");
		printf("Enter arguments like this: \n");
		printf("-i <dimX> -j <dimY> -k <dimK> \n");
		exit(EXIT_FAILURE);
	}

	// Size in bytes of input matrix
	size_t input_bytes = input_x * input_y * sizeof(unsigned int);

	// Size in bytes of mask
	size_t mask_bytes = mask_size * sizeof(unsigned int);

	// Size in bytes of output matrix (result)
	size_t output_bytes = input_x * input_y * sizeof(unsigned int);

	// Allocate host memory for input matrix, mask, and output matrix
	unsigned int* host_input;
	unsigned int* host_mask;
	unsigned int* host_output;
	unsigned int* host_output_cpu;
	host_input = (unsigned int*)malloc(input_bytes);
	host_mask = (unsigned int*)malloc(mask_bytes);
	host_output = (unsigned int*)malloc(output_bytes);
	host_output_cpu = (unsigned int*)malloc(output_bytes);

	// Allocate device memory for input vector and bins
	unsigned int* device_input;
	unsigned int* device_output;
	checkCudaErrors(cudaMalloc((void**)&device_input, input_bytes));
	checkCudaErrors(cudaMalloc((void**)&device_output, output_bytes));

	// Initialize input matrix with ints between 0~15
	srand((unsigned int)time(NULL));		// Assigns seed to make random numbers change
	for (int i = 0; i < input_x * input_y; i++) {
		host_input[i] = 1;
		//host_input[i] = rand() % 16;
	}

	// Initialize mask with ints between 0~15
	srand((unsigned int)time(NULL));		// Assigns seed to make random numbers change
	for (int i = 0; i < mask_size; i++) {
		host_mask[i] = 1;
		//host_mask[i] = rand() % 16;
	}
	
	// Initialize output (result) matrix with 0s
	for (int i = 0; i < input_x * input_y; i++) {
		host_output[i] = 0;
	}
	for (int i = 0; i < input_x * input_y; i++) {
		host_output_cpu[i] = 0;
	}

	// Print input matrix
	printf("\nInput matrix:\n");
	for (int i = 0; i < input_x * input_y; i++) {
		printf("%d\t", host_input[i]);
		if ((i + 1) % input_x == 0) {		// At end of each row make a new line
			printf("\n");
		}
	}

	// Print mask
	printf("\nMask:\n");
	for (int i = 0; i < mask_size; i++) {
		printf("%d\t", host_mask[i]);
	}

	// Copy matrix values from host to device
	checkCudaErrors(cudaMemcpy(device_input, host_input, input_bytes, cudaMemcpyHostToDevice));		// dest, source, size in bytes, direction of transfer
	checkCudaErrors(cudaMemcpy(device_output, host_output, output_bytes, cudaMemcpyHostToDevice));		// dest, source, size in bytes, direction of transfer

	// Copy mask values from host to device constant memory
	checkCudaErrors(cudaMemcpyToSymbol(device_mask, host_mask, mask_bytes));

	// Set Grid and Block sizes
	int block_size = 16;									// Threads per block
	int grid_size = ceil(input_x * input_y / block_size) + 1;
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

	int nIter = 100;	// How many times to run kernel

	printf("\n\nStarting Convolution on GPU\n");

	// Launch kernel (repeat nIter times so we can obtain average run time)
	// Leave the kernel you want to use un-commented and comment out the rest
	for (int i = 0; i < nIter; i++) {
		Convolution_GPU << <dim_grid, dim_block >> > (device_input, device_output, input_x, input_y, mask_size);
	}

	printf("\n\GPU Convolution Complete\n");

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	float msecPerConvolution = msecTotal / nIter;
	printf("\nGPU Histogram Computation took %.3f msec\n", msecPerConvolution);
	printf("\nThreads per block = %d, Blocks per grid = %d, Total threads = %d\n", block_size, grid_size, block_size * grid_size);

	// Copy matrix values from device to host
	checkCudaErrors(cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost));

	//// Make sure bins from GPU have correct values
	//for (int i = 0; i < num_bins; i++) {
	//	host_bins[i] /= nIter;
	//}

	// Print GPU results
	printf("\nGPU Results: \n");
	for (int i = 0; i < input_x * input_y; i++) {
		printf("%d\t", host_output[i]);
		if ((i + 1) % input_x == 0) {		// At end of each row make a new line
			printf("\n");
		}
	}

	////Start CPU timer
	//double time_taken_cpu = 0.0;
	//clock_t begin_cpu = clock();

	//// Calculate histogram on CPU
	//printf("\nStarting Histogram Computation on CPU\n");
	//for (int i = 0; i < nIter; i++) {		// Repeat CPU computation same amount of times as GPU computation
	//	Histogram_CPU(host_input, input_size, bin_size, host_bins_cpu);
	//}
	//printf("\nCPU Histogram Computation Complete\n");

	//clock_t end_cpu = clock();
	//time_taken_cpu += ((double)(end_cpu - begin_cpu) / CLOCKS_PER_SEC * 1000) / nIter;	// in milliseconds
	//printf("\nCPU Histogram Computation took %.3f msec\n", time_taken_cpu);

	//// Make sure bins from CPU have correct values
	//for (int i = 0; i < num_bins; i++) {
	//	host_bins_cpu[i] /= nIter;
	//}

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

	//// Check if GPU and CPU histograms match
	//bool check = 0;
	//for (int i = 0; i < num_bins; i++) {			// For every value in the arrays
	//	if (host_bins[i] != host_bins_cpu[i]) {		// Check if they match and if not set a flag
	//		check = 1;
	//	}
	//}
	//if (check == 1) {
	//	printf("\nGPU and CPU histograms do not match!\n");
	//}
	//else {
	//	printf("\nGPU and CPU histograms match!\n");
	//}

	// Free memory in device
	cudaFree(device_input);
	cudaFree(device_output);

	// Free memory in host
	free(host_input);
	free(host_mask);
	free(host_output);
	free(host_output_cpu);
}
