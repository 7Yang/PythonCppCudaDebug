#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using namespace std;

__device__ void triple(float* x)
{
	float tmp = *x;
	*x        = tmp * 3;
}

__global__ inline void kernel(float* a, float* b, float* c, size_t n)
{
	const int32_t nidx = blockIdx.x * blockDim.x + threadIdx.x;
	if(nidx > n) return;

    triple(&a[nidx]);
	c[nidx]=a[nidx]+b[nidx];
}


__host__ void myAddcuda(float*a,float*b,float*c,size_t n)
{
	float *device_a, *device_b, *device_c;
	cudaMalloc((void**)&device_a, sizeof(float)*n);
	cudaMalloc((void**)&device_b, sizeof(float)*n);
	cudaMalloc((void**)&device_c, sizeof(float)*n);

	cudaMemcpy(device_a, a, sizeof(float)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, b, sizeof(float)*n, cudaMemcpyHostToDevice);

	size_t thread_count = 5;
	size_t block_count  = (n-1)/thread_count+1;
	kernel<<<block_count, thread_count>>>(device_a,device_b,device_c,n);
	cudaDeviceSynchronize();
	
	cudaMemcpy(c,device_c ,sizeof(float)*n , cudaMemcpyDeviceToHost);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
}

extern "C" void myround(int x)
{
	size_t N   = 17;
	float  a[] = {1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7};
	float  b[] = {1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7};
	float  c[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	a[0]       = (float)x;
	while(true)
	{
		myAddcuda(a,b,c,N);
		for(int i = 0;i<N;i++)
		{
			cout << c[i] << endl;
		}
	}
}
