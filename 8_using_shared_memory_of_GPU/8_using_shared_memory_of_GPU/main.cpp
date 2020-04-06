#include "cpu_to_gpu_mem.h"
#include "cpu_gpu_functions.h"

#include "kernel_gpu_add.cuh"
#include "cuda_runtime_api.h"

#define CONCURRENT_COUNT 1						// ayni anda asenkron calistirilacak islem miktari

void main()
{
	struct cpu_gpu_mem cgm[CONCURRENT_COUNT];
	const int numberCount = 101 * 1024 * 1024;	// allocation size

	for (int i = 0; i < CONCURRENT_COUNT; i++)
	{
		struct cpu_gpu_mem *cg = &cgm[i];

		cg->numberCount = numberCount;
		cg->numberCountTiny = 32;

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


	// L1 cache e bagimliligi test etmek icin, gpu_add fonk icinde fazladan bir dongu kullanildi ve cpu_p_tiny, gpu_p_tiny adres alanlari test icin olusturuldu..
	
	
	// Project -> Properties -> CUDA C/C++ -> Command line -> -Xptxas -dlcm=ca yapilirsa cache all yani tum cache seviyerleri kullanilacak (L1, L2 , Device memory ..)
	// Project -> Properties -> CUDA C/C++ -> Command line -> -Xptxas -dlcm=cg yapilirsa cache global yani L2 cache kullanilacak, L1 cache seviyesi kapatilacak ve program cogunlukla L2 cache e bagimli olacaktir (L2, Device memory ..)
	// L1 cache kapatimak istensede, L1 cache genellikle tamamen kapatilamaz, bundan dolayi bir miktar L1 cache uzerinde program bagimli olacaktir..
	// L2 cache ise globah cache dir, kapatilamaz.

}