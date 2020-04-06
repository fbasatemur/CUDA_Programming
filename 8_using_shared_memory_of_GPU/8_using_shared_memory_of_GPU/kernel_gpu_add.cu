#pragma once

#ifdef  __INTELLISENSE__		// syncthreads(), bir CUDA fonk oldugu icin vs INTELLISENSE in hata vermemesi icin
void __syncthreads();
#endif //  __INTELLISENSE__

#include "device_launch_parameters.h"

#include "cpu_to_gpu_mem.h"
#include "kernel_gpu_add.cuh"

#include "device_functions.h"

__global__ void gpu_add(int* big_set_numbers, const int big_set_count, int* tiny_set_numbers,const int tiny_set_count)		// __global__ prefix i vscc tarafindan anlasilmaz ve bu fonksiyonu nvcc compile edecektir
{
	extern __shared__ int tiny_shared[];		// shared memory uzerinde depolanacak array.. toplam shared memory alani block sayisina bolunerek, her dilim bir block icin tahsis edilir..

	int tidX = threadIdx.x;
	
	if (tidX < tiny_set_count)					// threadid, tiny_set_count tan kucuk oldudgu surece extern yapili tiny_shared i doldur..
	{
		tiny_shared[tidX] = tiny_set_numbers[tidX];
	}											
								// blockDim sayisi, tiny_set_count tan fazla olabilir ve fazlalik thread ler sonraki satirlara gecebilir..
	__syncthreads();			// tum thread lerin bu satira gelmesi beklenir.. Yani __syncthreads() bir bariyer gorevi gorur
								// tum thread ler __syncthreads() e gelmeden, hic bir thread sonraki satira gecemez..
	// tum thread lerin shread memory uzerinde tiny_shared i doldurmasindan sonra shared memory alanina, tiny_shared icin erisim yap

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id < big_set_count)
	{
		int total = big_set_numbers[id];
		for (int i = 0; i < tiny_set_count; i++)
		{
			total += tiny_shared[i];
		}

		big_set_numbers[id] *= total;
	}
}

void cpu_gpu_execute(struct cpu_gpu_mem* cgm)						// bu fonksiyonda ise vscc gpu_add e kadar compile eder; gpu_add fonksiyonlarini ise nvcc compile edecektir..
{
	int numberCount = cgm->numberCount;
	int numberCountTiny = cgm->numberCountTiny;

	int blockDim = 64;
	int gridDim = (numberCount + blockDim - 1) / blockDim;


	int sharedMemorySize = numberCountTiny * sizeof(int);			// shared memory de depolanacak array in toplam boyutu

	gpu_add << < gridDim, blockDim, sharedMemorySize, cgm->stream >> > ((int*)cgm->gpu_p, numberCount, (int*)cgm->gpu_p_tiny, numberCountTiny);		// gpu_add fonksiyonunu yalnizca .cu uzantili dosyada cagirabiliriz.
																											// sebebi bu fonksiyonlari vscc degil de nvcc in compile etmesini saglamak icindir 
																											// stream yapisini kullanabilmek icin GPU fonksiyonuna parametre olarak olusturulan streamler verilmelidir.
}