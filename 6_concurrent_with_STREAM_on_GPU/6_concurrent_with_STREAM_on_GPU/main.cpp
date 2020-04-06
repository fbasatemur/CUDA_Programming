#include "cpu_to_gpu_mem.h"
#include "cpu_gpu_functions.h"

#include "kernel_gpu_add.cuh"
#include "cuda_runtime_api.h"

#define CONCURRENT_COUNT 2						// ayni anda asenkron calistirilacak islem miktari

void main()
{
	struct cpu_gpu_mem cgm[CONCURRENT_COUNT];
	const int numberCount = 101 * 1024 * 1024;	// allocation size

	for (int i = 0; i < CONCURRENT_COUNT; i++)
	{
		struct cpu_gpu_mem *cg = &cgm[i];

		cg->numberCount = numberCount;

		cudaError_t result = cudaStreamCreate(&cg->stream);		// Asenkron islem icin stream yarat
		assert(result == cudaSuccess);

		cpu_gpu_alloc(cg);
		cpu_gpu_set_numbers(cg);
		cpu_gpu_pin(cg);
	}

	for (int i = 0; i < CONCURRENT_COUNT; i++)
	{
		struct cpu_gpu_mem* cg = &cgm[i];

		cpu_gpu_host_to_dev(cg);		// Host to device ya da dev to host islemlerinde data akisi farkli kanallar uzerinden yapilir..Ancak ayni kanallarda data aktariminda bir onceki aktarimin bitmesi beklenir
		cpu_gpu_execute(cg);			// Yani dev to host yaparken ayni anda host to dev fonk hiz kaybi yasamadan calistirilabilir.. Ancak bir host_to_dev bitmeden bir sonraki host_to dev aktarilanmaz
		cpu_gpu_dev_to_host(cg);		// Asenkron islemi yapabilmek icin Stream olusturmak zorundayiz. Stream, mem kopyalama ve GPU fonk kullanildigi yerlerde parametre verilir.
	}
	
	for (int i = 0; i < CONCURRENT_COUNT; i++)
	{
		struct cpu_gpu_mem* cg = &cgm[i];

		cpu_gpu_unpin(cg);
		cpu_gpu_free(cg);

		cudaStreamDestroy(cg->stream);							// Son olarak olusturulan stream, yok edilmelidir.
	}

	cudaError_t result = cudaDeviceSynchronize();				// Asenkron islemler yapildigindan, hatanin tespiti icin son kez senkron yapilir
	assert(result == cudaSuccess);

	//cpu_gpu_print_results(&cg);
}