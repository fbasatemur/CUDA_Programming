
#include "cpu_gpu_functions.h"
#include "cuda_runtime.h"			// cudaError_t

#define PINLIMIT 100*1024*1024		// 100 MB uzeri sabitlenmis(pin lenmis) bellek uzerinden kopyalama yapsin

int get_allocation_size(const int numberCount)
{
	return numberCount * sizeof(int);
}

void cpu_gpu_alloc(struct cpu_gpu_mem* cgm)
{
	int allocationSize = get_allocation_size(cgm->numberCount);
	int allocationSizeTiny = get_allocation_size(cgm->numberCountTiny);

	cgm->cpu_p = malloc(allocationSize);
	cgm->cpu_p_tiny= malloc(allocationSizeTiny);

	cudaError_t result = cudaMalloc(&cgm->gpu_p, allocationSize);
	assert(result == cudaSuccess);

	result = cudaMalloc(&cgm->gpu_p_tiny, allocationSizeTiny);
	assert(result == cudaSuccess);
}

void cpu_gpu_free(struct cpu_gpu_mem* cgm)
{
	cudaError_t result = cudaFree(cgm->gpu_p);
	assert(result == cudaSuccess);

	free(cgm->cpu_p);


	result = cudaFree(cgm->gpu_p_tiny);
	assert(result == cudaSuccess);

	free(cgm->cpu_p_tiny);
}

void cpu_gpu_set_numbers(struct cpu_gpu_mem* cgm)
{
	int* cpu_int32_p = (int*)cgm->cpu_p;

	for (int i = 0; i < cgm->numberCount; i++)
		cpu_int32_p[i] = i;

	int* cpu_int32_p_tiny = (int*)cgm->cpu_p_tiny;

	for (int i = 0; i < cgm->numberCountTiny; i++)
		cpu_int32_p_tiny[i] = i;

}

void cpu_gpu_pin(struct cpu_gpu_mem* cgm)
{
	int allocationSize = get_allocation_size(cgm->numberCount);

	if (allocationSize > PINLIMIT)
	{
		cudaError_t result = cudaHostRegister(cgm->cpu_p, allocationSize, 0);	// cudaHostRegister ile yalnizca cudaMemcpy yapicagimiz zaman bellegi sabitleyebiliriz(pinleriz) ..	
		assert(result == cudaSuccess);
	}
}

void cpu_gpu_unpin(struct cpu_gpu_mem* cgm)
{
	int allocationSize = get_allocation_size(cgm->numberCount);

	if (allocationSize > PINLIMIT)
	{
		cudaError_t result = cudaHostUnregister(cgm->cpu_p);					// pin lenmis bellek bolgesi copy sonrasi, diger processlerin kullanabilmesi icin serbest birakilir.
		assert(result == cudaSuccess);
		// boylece bellek uzun sure pin lenmis olarak kalmicak, yalnizca kopyalama yapilacagi zaman bellek alani sabitlenerek hiz kazanilacak..
	}
}
void cpu_gpu_mem_copy(struct cpu_gpu_mem* cgm, enum cudaMemcpyKind copyKind)
{
	int allocationSize = get_allocation_size(cgm->numberCount);
	int allocationSizeTiny = get_allocation_size(cgm->numberCountTiny);
	cudaError_t result;

	switch (copyKind)
	{
	case cudaMemcpyDeviceToHost:
		result = cudaMemcpyAsync(cgm->cpu_p, cgm->gpu_p, allocationSize, cudaMemcpyDeviceToHost, cgm->stream);// Stream yapisi oldugundan dolayi cudaMemcpy yerine asenkron yapilabilmesi icin cudaMemcpyAsync kullanilir
		assert(result == cudaSuccess);
		break;

	case cudaMemcpyHostToDevice:
		result = cudaMemcpyAsync(cgm->gpu_p, cgm->cpu_p, allocationSize, cudaMemcpyHostToDevice, cgm->stream);// Stream yapisi oldugundan dolayi cudaMemcpy yerine asenkron yapilabilmesi icin cudaMemcpyAsync kullanilir
		assert(result == cudaSuccess);

		result = cudaMemcpyAsync(cgm->gpu_p_tiny, cgm->cpu_p_tiny, allocationSizeTiny, cudaMemcpyHostToDevice, cgm->stream); 
		assert(result == cudaSuccess);
		break;

	default:
		abort();												// kosullar saglanmaz ise programi sonlandir
		break;
	}
}

void cpu_gpu_host_to_dev(struct cpu_gpu_mem* cgm)
{
	cpu_gpu_mem_copy(cgm, cudaMemcpyHostToDevice);				// cudaHostRegister kullanarak kopyalama yapma islemi fonk duzeyinde ve belirli bir limit ile gerceklendi
}

void cpu_gpu_dev_to_host(struct cpu_gpu_mem* cgm)
{
	cpu_gpu_mem_copy(cgm, cudaMemcpyDeviceToHost);				// cudaHostRegister kullanarak kopyalama yapma islemi fonk duzeyinde ve belirli bir limit ile gerceklendi
}

void cpu_gpu_print_results(struct cpu_gpu_mem* cgm)
{
	int* cpu_int32_p = (int*)cgm->cpu_p;

	for (int i = cgm->numberCount - 100; i < cgm->numberCount; i++)
		printf("%d \t %d \t\n", i, cpu_int32_p[i]);
}