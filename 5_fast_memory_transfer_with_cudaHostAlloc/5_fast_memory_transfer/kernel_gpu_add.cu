#include "device_launch_parameters.h"

#include "cpu_to_gpu_mem.h"
#include "kernel_gpu_add.cuh"

__global__ void gpu_add(int* gpu_numbers, const int numberCount)				// __global__ prefix i vscc tarafindan anlasilmaz ve bu fonksiyonu nvcc compile edecektir
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < numberCount)
		gpu_numbers[id] *= 2;

}

void cpu_gpu_execute(struct cpu_gpu_mem* cgm)									// bu fonksiyonda ise vscc gpu_add e kadar compile eder; gpu_add fonksiyonlarini ise nvcc compile edecektir..
{
	int numberCount = cgm->numberCount;

	int blockDim = 64;
	int gridDim = (numberCount + blockDim - 1) / blockDim;

	gpu_add << < gridDim, blockDim >> > ((int*)cgm->gpu_p, numberCount);		// gpu_add fonksiyonunu yalnizca .cu uzantili dosyada cagirabiliriz.
	gpu_add << < gridDim, blockDim >> > ((int*)cgm->gpu_p, numberCount);		// sebebi bu fonksiyonlari vscc degil de nvcc in compile etmesini saglamak icindir 
	gpu_add << < gridDim, blockDim >> > ((int*)cgm->gpu_p, numberCount);
}