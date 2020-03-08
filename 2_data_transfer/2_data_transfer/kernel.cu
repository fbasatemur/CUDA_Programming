#include<iostream>
#include<stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>

int number_count = 640;
const int allocation_size = number_count * sizeof(int);

void* cpu_p;
void* gpu_p;


void cpuSetNumbers()
{
	int* cpu_int32 = (int*)cpu_p;

	for (int i = 0; i < number_count; i++)
		cpu_int32[i] = i;

}

int cpuAlloc()							// return 1 -> failed 
{
	cpu_p = malloc(allocation_size);

	if (cpu_p != nullptr)
		return 0;
	return 1;
}

cudaError_t gpuAlloc()
{
	cudaError_t result = cudaMalloc(&gpu_p, allocation_size);
	return result;
}


cudaError_t cpuMemoryToGpuMemory()
{
	cudaError_t result = cudaMemcpy(gpu_p, cpu_p, allocation_size, cudaMemcpyHostToDevice);		// cpu memory to gpu memory
	return result;
}


cudaError_t gpuMemoryToCpuMemory()
{
	cudaError_t result = cudaMemcpy(cpu_p, gpu_p, allocation_size, cudaMemcpyDeviceToHost);		// gpu memory to cpu memory
	return result;
}

void cpuFree()
{
	free(cpu_p);
}

cudaError_t gpuFree()
{
	cudaError_t result = cudaFree(gpu_p);
	return result;
}

__global__ void gpuAdd(int* gpu_numbers)
{
	int threadId = threadIdx.x;

	gpu_numbers[threadId] *= 2;
}

void printCpuNumbers()
{
	int* cpu_int32 = (int*)cpu_p;

	for (size_t i = 0; i < number_count; i++) {
		printf("%d\t%d\n", i, cpu_int32[i]);
	}
}

void main()
{
	std::cout << cpuAlloc();
	cpuSetNumbers();

	std::cout << gpuAlloc();
	cpuMemoryToGpuMemory();

	// data process by gpu 

	gpuAdd << < 1, number_count >> > ((int*)gpu_p);
	cudaError_t result = cudaDeviceSynchronize();		// cudaDeviceSynchronize -> waited all threads finish
	assert(result == cudaSuccess);						// if it is result = 0, process successful  
		

	gpuMemoryToCpuMemory();

	printCpuNumbers();
	
	gpuFree();
	cpuFree();

	system("pause");
}