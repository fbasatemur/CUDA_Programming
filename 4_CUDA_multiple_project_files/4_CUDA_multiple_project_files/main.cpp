#include "cpu_to_gpu_mem.h"
#include "cpu_gpu_functions.h"

#include "kernel_gpu_add.cuh"

void main()
{
	const int numberCount = 1024 * 1024;	// allocation size
	struct cpu_gpu_mem cgm;

	cgm.numberCount = numberCount;

	cpu_gpu_alloc(&cgm);
	cpu_gpu_set_numbers(&cgm);

	cpu_gpu_host_to_dev(&cgm);
	cpu_gpu_execute(&cgm);
	cpu_gpu_dev_to_host(&cgm);

	cpu_gpu_print_results(&cgm);

	cpu_gpu_free(&cgm);

}