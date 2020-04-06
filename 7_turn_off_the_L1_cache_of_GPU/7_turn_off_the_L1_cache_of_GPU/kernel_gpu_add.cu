#include "device_launch_parameters.h"

#include "cpu_to_gpu_mem.h"
#include "kernel_gpu_add.cuh"

__global__ void gpu_add(int* big_set_numbers, const int big_set_count, int* tiny_set_numbers,const int tiny_set_count)				// __global__ prefix i vscc tarafindan anlasilmaz ve bu fonksiyonu nvcc compile edecektir
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < big_set_count)
	{
		int total = big_set_numbers[id];
		for (int i = 0; i < tiny_set_count; i++)
		{
			total += tiny_set_numbers[i];
		}

		big_set_numbers[id] *= total;
	}
}

void cpu_gpu_execute(struct cpu_gpu_mem* cgm)									// bu fonksiyonda ise vscc gpu_add e kadar compile eder; gpu_add fonksiyonlarini ise nvcc compile edecektir..
{
	int numberCount = cgm->numberCount;
	int numberCountTiny = cgm->numberCountTiny;

	int blockDim = 64;
	int gridDim = (numberCount + blockDim - 1) / blockDim;


	gpu_add << < gridDim, blockDim, 0, cgm->stream >> > ((int*)cgm->gpu_p, numberCount, (int*)cgm->gpu_p_tiny, numberCountTiny);		// gpu_add fonksiyonunu yalnizca .cu uzantili dosyada cagirabiliriz.
																											// sebebi bu fonksiyonlari vscc degil de nvcc in compile etmesini saglamak icindir 
																											// stream yapisini kullanabilmek icin GPU fonksiyonuna parametre olarak olusturulan streamler verilmelidir.
}