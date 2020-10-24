// CUDA Matrix Multiplication with Tiling Algorithm
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
#define SHARED_MEM_SIZE 16 * 16	// Dimensions of each block/tile

// Kernel function for GPU matrix multiplication
__global__ void Matrix_Multiplication(float * device_m, float * device_n, float * device_p, unsigned int rowDimM, unsigned int colDimM, unsigned int colDimN) {

	// Allocate shared memory (private for each thread block)
	__device__ __shared__ int M[SHARED_MEM_SIZE];
	__device__ __shared__ int N[SHARED_MEM_SIZE];

	// Calculate row and column indexes for each thread (location of the thread)
	int Row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int Col = (blockIdx.x * blockDim.x) + threadIdx.x;

	// Temporary sum of dot products
	float P_value = 0;
	int num_of_phases = ceil((float)colDimM / blockDim.x);		// Width of each block/tile, has to be at least 1

	// Slide tile across grid and load values into tile in shared memory
	for (int phase = 0; phase < num_of_phases; phase++) {
		/*
		* Row * Width			= which row we belong to
		* phase * blockDim.x	= which step of tile sliding we are at in x direction
		* threadIdx.x			= which thread (column) we are at 
		*/
		if (((phase * blockDim.x) + threadIdx.x < colDimM) && Row < rowDimM) {		// While within horizontal bounds of matrix M
			M[(threadIdx.y * blockDim.x) + threadIdx.x] = device_m[(Row * colDimM) + (phase * blockDim.x) + threadIdx.x];
		}
		else {
			M[(threadIdx.y * blockDim.x) + threadIdx.x] = 0;	// If outside of bounds of matrix M
		}
		/*
		* phase * blockDim.x * Width	= which step of tile sliding we are at in y direciton
		* threadIdx.y * Width			= which thread (row) we are at
		* Col							= which column we are at (constant)
		*/
		if (((phase * blockDim.y) + threadIdx.y < colDimM) && Col < colDimN) {		// While within vertical bounds of matrix N
			N[(threadIdx.y * blockDim.x) + threadIdx.x] = device_n[((phase * blockDim.x) + threadIdx.y) * colDimN + Col];
		}
		else {													// If outside of bounds of matrix N
			N[(threadIdx.y * blockDim.x) + threadIdx.x] = 0; 
		}

		// Barrier synchronization - lets all calculations finish before continuing
		__syncthreads();

		// Here is the actual matrix multiplication (dot products)
		// This is for each tile 
		for (int i = 0; i < blockDim.x; i++) {
			P_value += M[(threadIdx.y * blockDim.x) + i] * N[(i * blockDim.x) + threadIdx.x];
		}
		//Sync again so next phase can start
		__syncthreads();
	}
	// Write results to p matrix
	if ((Row < rowDimM) && (Col < colDimN)) {
		device_p[Row * colDimN + Col] = P_value;
	}
}

// Function to do matrix multiplication on CPU (to verify GPU calculations)
void CPU_Matrix_Multiplication(float *host_m, float *host_n, float *host_p_cpu, unsigned int rowDimM, unsigned int colDimM, unsigned int colDimN) {
	float temp_sum;
	for (int i = 0; i < rowDimM; i++) {		// For each row
		for (int j = 0; j < colDimN; j++) {		// For each column
			temp_sum = 0;
			for (int k = 0; k < colDimM; k++) {		// For each value
				temp_sum += host_m[i * colDimM + k] * host_n[k * colDimN + j];	// Calculate dot product
			}
			host_p_cpu[i * colDimN + j] = temp_sum;	// Transfer current values to p matrix
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
	if (argc == 5) {		// 5 arguments expected (filename, -i, rowDimM, colDimM, colDimN)
		printf("\n");
		printf("Matrix M has %s rows \n", argv[2]); 
		printf("Matrix M has %s columns \n\n", argv[3]);
		printf("Matrix N has %s rows \n", argv[3]);
		printf("Matrix N has %s columns \n", argv[4]);
	}
	else if (argc > 5) {
		printf("Too many arguments provided. \n");
		printf("Enter arguments like this: \n");
		printf("-i <rowDimA> <colDimA> <colDimB> \n");
		exit(EXIT_FAILURE);
	}
	else {
		printf("4 arguments expected. \n");
		printf("Enter arguments like this: \n");
		printf("-i <rowDimM> <colDimM> <colDimN> \n");
		exit(EXIT_FAILURE);
	}

	printf("\nStarting Matrix Multiplication on GPU\n");

	// Set matrix M dimensions
	int rowDimM_string;
	rowDimM_string = atoi(argv[2]);		// Convert string to int
	int rowDimM = rowDimM_string;
	
	int colDimM_string;
	colDimM_string = atoi(argv[3]);
	int colDimM = colDimM_string;

	// Size in bytes of matrix M
	size_t bytesM = rowDimM * colDimM * sizeof(float);

	// Set matrix N dimensions
	int rowDimN_string;
	rowDimN_string = atoi(argv[3]);
	int rowDimN = rowDimN_string;

	int colDimN_string;
	colDimN_string = atoi(argv[4]);
	int colDimN = colDimN_string;

	// Size in bytes of matrix N
	size_t bytesN = rowDimN * colDimN * sizeof(float);

	// Size in bytes of matrix P (result)
	size_t bytesP = rowDimM * colDimN * sizeof(float);

	// Allocate host memory for matrices
	float* host_m;
	float* host_n;
	float* host_p;
	float* host_p_cpu;

	host_m = (float*)malloc(bytesM);
	host_n = (float*)malloc(bytesN);
	host_p = (float*)malloc(bytesP);
	host_p_cpu = (float*)malloc(bytesP);

	// Allocate device memory for matrices
	float* device_m;
	float* device_n;
	float* device_p;

	cudaMalloc(&device_m, bytesM);
	cudaMalloc(&device_n, bytesN);
	cudaMalloc(&device_p, bytesP);

	// Initialize matrix M with values (on host)
	for (int i = 0; i < rowDimM * colDimM; i++) {
		host_m[i] = rand() % 100;		// random floats up to 100
	}
	// Initialize matrix N with values (on host)
	for (int i = 0; i < rowDimN * colDimN; i++) {
		host_n[i] = rand() & 100;
	}

	// Print input matrices (comment out unless wanted)
	//printf("\nMatrix M:\n");
	//for (int i = 0; i < rowDimM * colDimM; i++) {
	//	printf("%f\t", host_m[i]);
	//	if ((i + 1) % colDimM == 0) {	// At end of each row make new line
	//		printf("\n");
	//	}
	//} 
	//printf("\nMatrix N:\n");
	//for (int i = 0; i < rowDimN * colDimN; i++) {
	//	printf("%f\t", host_n[i]);
	//	if ((i + 1) % colDimN == 0) {
	//		printf("\n");
	//	}
	//}

	// Copy matrix values from host to device
	cudaMemcpy(device_m, host_m, bytesM, cudaMemcpyHostToDevice);		// dest, source, size in bytes, direction of transfer
	cudaMemcpy(device_n, host_n, bytesN, cudaMemcpyHostToDevice);

	// Set Grid and Block sizes
	int block_size = 16;		// Threads per block, also tile width
	dim3 DimBlock(block_size, block_size);
	dim3 DimGrid(ceil(colDimN / block_size) + 1, ceil(rowDimM / block_size) + 1);

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
		Matrix_Multiplication<<<DimGrid, DimBlock>>>(device_m, device_n, device_p, rowDimM, colDimM, colDimN);
	}

	printf("\nGPU Matrix Multiplication Complete\n");

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, stream));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	float msecPerMatrixMul = msecTotal / nIter;
	double flopsPerMatrixMul = 2.0 * static_cast<double>(colDimM) *
		static_cast<double>(rowDimM) *
		static_cast<double>(colDimN);
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
		(msecPerMatrixMul / 1000.0f);
	printf(
		"\nPerformance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops," \
		" WorkgroupSize= %u threads/block\n",
		gigaFlops,					// GFlop/s
		msecPerMatrixMul,			// Time (ms)
		flopsPerMatrixMul,			// Size (Flop/s per matrix mul)
		block_size * block_size);	// Thread per block

	// Copy matrix values from device to host
	cudaMemcpy(host_p, device_p, bytesP, cudaMemcpyDeviceToHost);

	// Print GPU result (comment out unless wanted)
	/*printf("\nMatrix P (GPU):\n");
	for (int i = 0; i < rowDimM * colDimN; i++) {
		printf("%f\t", host_p[i]);
		if ((i + 1) % colDimN == 0) {
			printf("\n");
		}
	}*/

	// Start timer for CPU calculation
	double time_taken_cpu = 0.0;
	clock_t begin_cpu = clock();

	// Do matrix multiplication on CPU to verify
	printf("\nStarting Matrix Multiplication on CPU\n");
	CPU_Matrix_Multiplication(host_m, host_n, host_p_cpu, rowDimM, colDimM, colDimN);
	printf("\nCPU Matrix Multiplication Complete\n");

	clock_t end_cpu = clock();

	time_taken_cpu += (double)(end_cpu - begin_cpu) / CLOCKS_PER_SEC * 1000;	// in milliseconds

	printf("\nCPU Matrix Multiplication took %f millisecond(s) to execute. \n", time_taken_cpu);

	// Print CPU result (comment out unless wanted)
	/*printf("\nMatrix P (CPU):\n");
	for (int i = 0; i < rowDimM * colDimN; i++) {
		printf("%f\t", host_p_cpu[i]);
		if ((i + 1) % colDimN == 0) {
			printf("\n");
		}
	}*/

	// Check if GPU and CPU calculations match
	bool check = 0;
	for (int i = 0; i < rowDimM * colDimN; i++) {	// For every value in the arrays
		if (host_p[i] != host_p_cpu[i]) {			// Check if they match and if not set a flag
			check = 1;
		}
	}
	if (check == 1) {
		printf("\nGPU and CPU matrix multiplications do not match!\n");
	}
	else {
		printf("\nGPU and CPU matrix multiplications match!\n");
	}

	// Free memory in device
	cudaFree(device_m);
	cudaFree(device_n);
	cudaFree(device_p);

	// Free memory in host
	free(host_m);
	free(host_n);
	free(host_p);
	free(host_p_cpu);
}