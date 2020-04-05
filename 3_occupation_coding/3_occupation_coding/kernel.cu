#include<iostream>
#include<stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>

int number_count = 500 * 1024 * 1024;						// Allocation yapilacak int miktari
const int allocation_size = number_count * sizeof(int);		// number_count * 4 B

void* cpu_p;
void* gpu_p;



void cpuAlloc()												// allocation_size kadar RAM alani tahsis et
{
	cpu_p = malloc(allocation_size);
	assert(nullptr != cpu_p);
}

void gpuAlloc()												// allocation_size kadar GPU memory tahsis et
{
	cudaError_t result = cudaMalloc(&gpu_p, allocation_size);
	assert(result == cudaSuccess);
}



void cpuMemoryToGpuMemory()									// cpu memory alanini (RAM), gpu memory alanina kopyala
{
	cudaError_t result = cudaMemcpy(gpu_p, cpu_p, allocation_size, cudaMemcpyHostToDevice);		// cpu memory to gpu memory
	assert(result == cudaSuccess);
}

void gpuMemoryToCpuMemory()									// gpu memory alanini, cpu memory alanina kopyala
{
	cudaError_t result = cudaMemcpy(cpu_p, gpu_p, allocation_size, cudaMemcpyDeviceToHost);		// gpu memory to cpu memory
	assert(result == cudaSuccess);
}


void cpuSetNumbers()										// cpu bellek alanina, number_count kadar sayi setle
{
	int* cpu_int32 = (int*)cpu_p;

	for (int i = 0; i < number_count; i++)
		cpu_int32[i] = i;

}

__global__ void gpuAdd(int* gpu_numbers)					// Paralel islemlenecek kisim, nvcc tarafindan burada compiler edilir
{
	//	int threadIndexOfTheThread = threadIdx.x;								// anlik thread index
	//	int blockIndexOfTheThread = blockIdx.x;									// anlik block index

	//	int threadCountInOneBlock = blockDim.x;									// bir bloktaki toplam thread sayisi
	//	int blockCountInThisKernel = gridDim.x;									// toplam block sayisi
	//	
	//	int id = blockIndexOfTheThread * threadCountInOneBlock + threadIndexOfTheThread;

	//	printf("%d \t %d \t  %d \t  %d \t  %d \t \n", id, threadIndexOfTheThread, blockIndexOfTheThread, blockCountInThisKernel, threadCountInOneBlock);

	
	// bir block sonlandiginda, thread index tekrar sifirlanacaktir.Bu ise veri karmasasina neden olabilir.
	int id = blockIdx.x * blockDim.x + threadIdx.x;			// Bunu engellemek icin index degeri duzenlenir

	gpu_numbers[id] *= 2;
}



void printCpuNumbers()										
{
	int* cpu_int32 = (int*)cpu_p;

	for (size_t i = number_count - 100; i < number_count; i++)		// son 100 degeri yazdir 
	{		
		printf("%d\t%d\n", i, cpu_int32[i]);
	}
}


void cpuFree()														// cpu memory serbest birak
{
	free(cpu_p);
}

void gpuFree()														// gpu memory serbest birak
{
	cudaError_t result = cudaFree(gpu_p);
	assert(result == cudaSuccess);
}

void main()
{
	cpuAlloc();
	cpuSetNumbers();

	gpuAlloc();
	cpuMemoryToGpuMemory();

	// GPU bellegi uzerinden paralel veri islemleme yapiliyor..

	int blockDim = 64;									// Bir bloktaki toplam thread sayisi
	int gridDim = number_count / blockDim;				// toplam block sayisi = toplam thread sayisi / blockDim

	gpuAdd <<< gridDim, blockDim >> > ((int*)gpu_p);
	// GPU uzerinde tum islemler asenkron olarak yapilacaktir...
	cudaError_t result = cudaDeviceSynchronize();		// cudaDeviceSynchronize ile tum islemlerin bitmesini bekleriz.. 
	assert(result == cudaSuccess);						// if it is result = 0, process successful  


	gpuMemoryToCpuMemory();								// Cpu memory alanina, Gpu bellek alaninda islemlenen tum degerler aktarilir

	printCpuNumbers();

	gpuFree();
	cpuFree();

	getchar();
}