#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"
#include <stdlib.h>
#include <iostream>

const long allocation_size = 1 * 1024 * 1024 * 1024;	// 1GB allocation size

void* cpu_p;
void* gpu_p;

void cpuAlloc()											// cpu memory allocation
{
	cpu_p = malloc(allocation_size);
}

cudaError_t gpuAlloc()
{
	cudaError_t result = cudaMalloc(&gpu_p, allocation_size);
	return result;
}

cudaError_t gpuFree()
{
	cudaError_t result = cudaFree(gpu_p);
	return result;
}

void main()
{

	cpuAlloc();
	std::cout << gpuAlloc();							// result = 0 -> allocation success

	system("pause");

	try
	{
		std::cout << gpuFree();							// result = 0 -> flush success
		free(cpu_p);
	}
	catch (const std::exception & error)
	{
		std::cout << error.what();
	}

}