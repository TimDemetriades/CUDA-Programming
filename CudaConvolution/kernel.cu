// CUDA Convolution
// Tim Demetriades
// CPE 810 - GPU & Multicore Programming
// Professor Feng
// Stevens Institute of Technology

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>	// for CPU timer

#define MAX_MASK_SIZE 128							// Set max mask size (arbitrary)
__constant__ int device_mask[MAX_MASK_SIZE];		// Initialize mask as constant memory

// Kernel function for Convolution
// First mask runs through each row of the input matrix
/*
* device_input = input matrix
* device_buffer = buffer matrix (holds input for second kernel)
* input_x = number of columns of input matrix
* input_y = number of rows of input matrix
* mask_size = width of mask
*/
__global__ void Convolution_GPU_X(unsigned int* device_input, unsigned int* device_buffer, unsigned int input_x, unsigned int input_y, unsigned int mask_size) {

	// Calculate row and column indexes for each thread (location of the thread)
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Start point for rows and columns
	int row_start = row - (mask_size / 2);

	// Temp value for result accumulation
	int temp = 0;

	// Convolution
	for (int i = 0; i < mask_size; i++) {
		if (((row_start + i) >= 0) && ((row_start + i) < input_x)) {	// Only do multiplication if we are not over ghost cells
			temp += device_input[row_start + i] * device_mask[i];
		}
	}

	// Write results to buffer matrix
	if ((row < input_y) && (col < input_x)) {		// Make sure we're in bounds of matrix
		device_buffer[col * input_y + row] = temp;
	}
}
// Then mask runs through each column of the input matrix
/*
* device_buffer = results from last kernel used as input here
* device_output = output results matrix
* input_x = number of columns of input matrix
* input_y = number of rows of input matrix
* mask_size = width of mask
*/
__global__ void Convolution_GPU_Y(unsigned int* device_buffer, unsigned int* device_output, unsigned int input_x, unsigned int input_y, unsigned int mask_size) {

	// Calculate row and column indexes for each thread (location of the thread)
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Start point for rows and columns
	int col_start = col - (mask_size / 2);

	// Temp value for result accumulation
	int temp = 0;

	// Convolution
	for (int i = 0; i < mask_size; i++) {
		if (((col_start + i) >= 0) && ((col_start + i) < input_y)) {	// Only do multiplication if we are not over ghost cells
			temp += device_buffer[col_start + i] * device_mask[i];
		}
	}

	// Write results to output matrix
	if ((row < input_y) && (col < input_x)) {		// Make sure we're in bounds of matrix
		device_output[row * input_x + col] = temp;
	}
}

// Convolution on CPU
// First mask runs through each row of the input matrix
/*
* host_input = input matrix
* host_mask = mask values
* host_buffer_cpu = buffer matrix (holds input for second kernel)
* input_x = number of columns of input matrix
* input_y = number of rows of input matrix
* mask_size = width of mask
*/
void Convolution_CPU_X(unsigned int* host_input, unsigned int* host_mask, unsigned int* host_buffer_cpu, unsigned int input_x, unsigned int input_y, unsigned int mask_size) {
	int position, temp;		// position holds current position, temp holds current convolution sum
	for (int row = 0; row < input_x; row++) {		// For each row in input matrix
		for (int col = 0; col < input_y; col++) {	// And each column in input matrix
			position = row - (mask_size / 2);		// Set position
			temp = 0;		// Reset temp
			for (int i = 0; i < mask_size; i++) {	// For each value in mask
				if ((position + i >= 0) && (position + i < input_x)) {		// Only do multiplication if we are not over ghost cells
					temp += host_input[position + i] * host_mask[i];
				}
			}
			host_buffer_cpu[col * input_y + row] = temp;	// Write results to output matrix
		}
	}
}
// Then mask runs through each column of the input matrix
/*
* host_buffer_cpu = results from last kernel used as input here
* host_mask = mask values
* host_output_cpu = output results matrix
* input_x = number of columns of input matrix
* input_y = number of rows of input matrix
* mask_size = width of mask
*/
void Convolution_CPU_Y(unsigned int* host_buffer_cpu, unsigned int* host_mask, unsigned int* host_output_cpu, unsigned int input_x, unsigned int input_y, unsigned int mask_size) {
	int position, temp;		// position holds current position, temp holds current convolution sum
	for (int row = 0; row < input_x; row++) {		// For each row in input matrix
		for (int col = 0; col < input_y; col++) {	// And each column in input matrix
			position = row - (mask_size / 2);		// Set position
			temp = 0;		// Reset temp
			for (int i = 0; i < mask_size; i++) {	// For each value in mask
				if ((position + i >= 0) && (position + i < input_y)) {		// Only do multiplication if we are not over ghost cells
					temp += host_buffer_cpu[position + i] * host_mask[i];
				}
			}
			host_output_cpu[col * input_y + row] = temp;	// Write results to output matrix
		}
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

	// Size in bytes of buffer matrix (output matrix of first kernel and input of second)
	size_t buffer_bytes = input_x * input_y * sizeof(unsigned int);

	// Size in bytes of output matrix (result)
	size_t output_bytes = input_x * input_y * sizeof(unsigned int);

	// Allocate host memory for input matrix, mask, buffer, and output matrix
	unsigned int* host_input;
	unsigned int* host_mask;
	unsigned int* host_buffer;
	unsigned int* host_output;
	unsigned int* host_buffer_cpu;
	unsigned int* host_output_cpu;
	host_input = (unsigned int*)malloc(input_bytes);
	host_mask = (unsigned int*)malloc(mask_bytes);
	host_buffer = (unsigned int*)malloc(buffer_bytes);
	host_output = (unsigned int*)malloc(output_bytes);
	host_buffer_cpu = (unsigned int*)malloc(output_bytes);
	host_output_cpu = (unsigned int*)malloc(output_bytes);

	// Allocate device memory for input matrix, buffer, and output matrix
	unsigned int* device_input;
	unsigned int* device_buffer;
	unsigned int* device_output;
	checkCudaErrors(cudaMalloc((void**)&device_input, input_bytes));
	checkCudaErrors(cudaMalloc((void**)&device_buffer, buffer_bytes));
	checkCudaErrors(cudaMalloc((void**)&device_output, output_bytes));

	// Initialize input matrix with ints between 0~15
	srand((unsigned int)time(NULL));		// Assigns seed to make random numbers change
	for (int i = 0; i < input_x * input_y; i++) {
		host_input[i] = 1;		// For testing purposes (comment below line out when using)
		//host_input[i] = rand() % 16;
	}

	// Initialize mask with ints between 0~15
	srand((unsigned int)time(NULL));		// Assigns seed to make random numbers change
	for (int i = 0; i < mask_size; i++) {
		host_mask[i] = 1;		// For testing purposes (comment below line out when using)
		//host_mask[i] = rand() % 16;
	}
	
	// Initialize output (result) and buffer matrix with 0s
	for (int i = 0; i < input_x * input_y; i++) {
		host_buffer[i] = 0;
	}
	for (int i = 0; i < input_x * input_y; i++) {
		host_output[i] = 0;
	}
	for (int i = 0; i < input_x * input_y; i++) {
		host_buffer_cpu[i] = 0;
	}
	for (int i = 0; i < input_x * input_y; i++) {
		host_output_cpu[i] = 0;
	}

	//// Print input matrix
	//printf("\nInput matrix:\n");
	//for (int i = 0; i < input_x * input_y; i++) {
	//	printf("%d\t", host_input[i]);
	//	if ((i + 1) % input_x == 0) {		// At end of each row make a new line
	//		printf("\n");
	//	}
	//}

	//// Print mask
	//printf("\nMask:\n");
	//for (int i = 0; i < mask_size; i++) {
	//	printf("%d\t", host_mask[i]);
	//}

	// Copy matrix values from host to device
	checkCudaErrors(cudaMemcpy(device_input, host_input, input_bytes, cudaMemcpyHostToDevice));			// dest, source, size in bytes, direction of transfer
	checkCudaErrors(cudaMemcpy(device_buffer, host_buffer, buffer_bytes, cudaMemcpyHostToDevice));		// dest, source, size in bytes, direction of transfer
	checkCudaErrors(cudaMemcpy(device_output, host_output, output_bytes, cudaMemcpyHostToDevice));		// dest, source, size in bytes, direction of transfer

	// Copy mask values from host to device constant memory
	checkCudaErrors(cudaMemcpyToSymbol(device_mask, host_mask, mask_bytes));							// dest, source, size in bytes

	// Set Grid and Block sizes
	int block_size = 16;									// Thread width
	int grid_size_x = ceil(input_x / block_size) + 1;
	int grid_size_y = ceil(input_y / block_size) + 1;
	dim3 dim_block(block_size, block_size);
	dim3 dim_grid(grid_size_x, grid_size_y);

	// Record the start event (for timing GPU calculations)
	cudaStream_t stream;
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	checkCudaErrors(cudaStreamSynchronize(stream));
	checkCudaErrors(cudaEventRecord(start, stream));

	int nIter = 100;	// How many times to run kernel

	printf("\nStarting Convolution on GPU\n");

	// Launch kernel (repeat nIter times so we can obtain average run time)
	// Leave the kernel you want to use un-commented and comment out the rest
	for (int i = 0; i < nIter; i++) {
		Convolution_GPU_X << <dim_grid, dim_block >> > (device_input, device_buffer, input_x, input_y, mask_size);
		Convolution_GPU_Y << <dim_grid, dim_block >> > (device_buffer, device_output, input_x, input_y, mask_size);
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
	double flopsPerConvolution = 2.0 * static_cast<double>(input_x) *
		static_cast<double>(input_y) *
		static_cast<double>(mask_size);
	double gigaFlops = (flopsPerConvolution * 1.0e-9f) /
		(msecPerConvolution / 1000.0f);
	printf(
		"\nPerformance = %.2f GFlop/s, Time = %.3f msec, Size = %.0f operations per convolution\n",
		gigaFlops,					// GFlop/s
		msecPerConvolution,			// Time (ms)
		flopsPerConvolution);		// Size (floating-point operations per convolution)
	printf("\nThreads per block = %d, Blocks per grid x = %d, Blocks per grid y = %d, Total threads = %d\n", block_size * block_size, grid_size_x, grid_size_y, (block_size * block_size) * (grid_size_x * grid_size_y));

	// Copy matrix values from device to host
	checkCudaErrors(cudaMemcpy(host_buffer, device_buffer, output_bytes, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost));

	//// Print GPU results
	//printf("\nGPU Buffer Results: \n");
	//for (int i = 0; i < input_x * input_y; i++) {
	//	printf("%d\t", host_buffer[i]);
	//	if ((i + 1) % input_x == 0) {		// At end of each row make a new line
	//		printf("\n");
	//	}
	//}

	//printf("\n\nGPU Results: \n");
	//for (int i = 0; i < input_x * input_y; i++) {
	//	printf("%d\t", host_output[i]);
	//	if ((i + 1) % input_x == 0) {		// At end of each row make a new line
	//		printf("\n");
	//	}
	//}

	//Start CPU timer
	double time_taken_cpu = 0.0;
	clock_t begin_cpu = clock();

	// Calculate histogram on CPU
	printf("\nStarting Convolution on CPU\n");
	for (int i = 0; i < nIter; i++) {		// Repeat CPU computation same amount of times as GPU computation
		Convolution_CPU_X(host_input, host_mask, host_buffer_cpu, input_x, input_y, mask_size);
		Convolution_CPU_Y(host_buffer_cpu, host_mask, host_output_cpu, input_x, input_y, mask_size);
	}
	printf("\nCPU Convolution Complete\n");

	clock_t end_cpu = clock();
	time_taken_cpu += ((double)(end_cpu - begin_cpu) / CLOCKS_PER_SEC * 1000) / nIter;	// in milliseconds

	double gigaFlops_CPU = (flopsPerConvolution * 1.0e-9f) / (time_taken_cpu / 1000.0f);
	printf(
		"\nPerformance = %.2f GFlop/s, Time = %.3f msec, Size = %.0f operations per convolution\n",
		gigaFlops_CPU,				// GFlop/s
		time_taken_cpu,				// Time (ms)
		flopsPerConvolution);		// Size (floating-point operations per convolution)

	//// Print CPU results
	//printf("\nCPU Buffer Results: \n");
	//for (int i = 0; i < input_x * input_y; i++) {
	//	printf("%d\t", host_buffer_cpu[i]);
	//	if ((i + 1) % input_x == 0) {		// At end of each row make a new line
	//		printf("\n");
	//	}
	//}
	//printf("\nCPU Results: \n");
	//for (int i = 0; i < input_x * input_y; i++) {
	//	printf("%d\t", host_output_cpu[i]);
	//	if ((i + 1) % input_x == 0) {		// At end of each row make a new line
	//		printf("\n");
	//	}
	//}

	// Check if GPU and CPU results match
	bool check = 0;
	for (int i = 0; i < input_x * input_y; i++) {			// For every value in the matrix
		if (host_output[i] != host_output_cpu[i]) {			// Check if they match and if not set a flag
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
	cudaFree(device_buffer);
	cudaFree(device_output);

	// Free memory in host
	free(host_input);
	free(host_mask);
	free(host_buffer);
	free(host_output);
	free(host_buffer_cpu);
	free(host_output_cpu);
}