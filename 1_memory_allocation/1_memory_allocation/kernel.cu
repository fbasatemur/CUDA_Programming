#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "stdio.h"
#include <stdlib.h>
#include <iostream>

const long allocation_size = 1 * 1024 * 1024 * 1024;

void* cpu_p;
void* gpu_p;

void cpuAlloc()
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
	std::cout << gpuAlloc();		// return 0 -> true

	system("pause");

	try
	{
		gpuFree();
		free(cpu_p);
	}
	catch (const std::exception & error)
	{
		std::cout << error.what();
	}

}