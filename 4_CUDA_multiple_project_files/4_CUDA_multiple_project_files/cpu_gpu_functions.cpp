
#include "cpu_gpu_functions.h"
#include "cuda_runtime.h"		// cudaError_t

int get_allocation_size(const int numberCount)
{
	return numberCount * sizeof(int);
}

void cpu_gpu_alloc(struct cpu_gpu_mem* cgm)
{
	int allocationSize = get_allocation_size(cgm->numberCount);

	cgm->cpu_p = malloc(allocationSize);

	cudaError_t result = cudaMalloc(&cgm->gpu_p, allocationSize);
	assert(result == cudaSuccess);
}

void cpu_gpu_free(struct cpu_gpu_mem* cgm)
{
	cudaError_t result = cudaFree(cgm->gpu_p);
	assert(result == cudaSuccess);

	free(cgm->cpu_p);
}

void cpu_gpu_set_numbers(struct cpu_gpu_mem* cgm)
{
	int* cpu_int32_p = (int*)cgm->cpu_p;

	for (int i = 0; i < cgm->numberCount; i++)
		cpu_int32_p[i] = i;

}

void cpu_gpu_host_to_dev(struct cpu_gpu_mem* cgm)
{
	cudaError_t result = cudaMemcpy(cgm->gpu_p, cgm->cpu_p, get_allocation_size(cgm->numberCount), cudaMemcpyHostToDevice);
	assert(result == cudaSuccess);
}

void cpu_gpu_dev_to_host(struct cpu_gpu_mem* cgm)
{
	cudaError_t result = cudaMemcpy(cgm->cpu_p, cgm->gpu_p, get_allocation_size(cgm->numberCount), cudaMemcpyDeviceToHost);
	assert(result == cudaSuccess);
}

void cpu_gpu_print_results(struct cpu_gpu_mem* cgm)
{
	int* cpu_int32_p = (int*)cgm->cpu_p;

	for (int i = cgm->numberCount - 100; i < cgm->numberCount; i++)
		printf("%d \t %d \t\n", i, cpu_int32_p[i]);
}