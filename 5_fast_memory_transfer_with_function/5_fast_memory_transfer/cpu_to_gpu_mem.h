#pragma once

struct cpu_gpu_mem {
	void* gpu_p;		// gpu pointer
	void* cpu_p;		// cpu pointer
	int numberCount;	// allocation size
};