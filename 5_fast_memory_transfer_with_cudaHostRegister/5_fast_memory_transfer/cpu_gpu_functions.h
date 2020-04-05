#pragma once


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "cpu_to_gpu_mem.h"

void cpu_gpu_alloc(struct cpu_gpu_mem* cgm_p);			// memory allocation
void cpu_gpu_free(struct cpu_gpu_mem* cgm_p);			// memory free

void cpu_gpu_set_numbers(struct cpu_gpu_mem* cgm_p);

void cpu_gpu_host_to_dev(struct cpu_gpu_mem* cgm_p);	// memory address is copy host to device
void cpu_gpu_dev_to_host(struct cpu_gpu_mem* cgm_p);	// memory address is copy device to host

void cpu_gpu_print_results(struct cpu_gpu_mem* cgm_p);


