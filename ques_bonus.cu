#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <algorithm>
#include <stdio.h>
#include "utils.h"
#define MAX_THREADS_PER_BLOCK 512

__global__ void block_sum(const unsigned int * const gpu_in,unsigned int * const gpu_out,unsigned int * const gpu_sum,const size_t n);
__global__ void incrementeement( unsigned int * const d_array,const unsigned int * const d_incr);


__global__ void block_sum(const unsigned int * const gpu_in,unsigned int * const gpu_out,unsigned int * const gpu_sum,const size_t n)
{
	extern __shared__ unsigned int smem[];
	const size_t bx = blockIdx.x * blockDim.x;
	const size_t tx = threadIdx.x;
	const size_t px = bx + tx;
	int offset = 1;
	smem[2*tx] = gpu_in[2*px];
	smem[2*tx+1] = gpu_in[2*px+1];
	
	for (int d = n >> 1; d > 0; d >>= 1)
	{
	__syncthreads();
		if (tx < d)
		{
		int ai = offset * (2*tx+1) - 1;
		int bi = offset * (2*tx+2) - 1;
		smem[bi] += smem[ai];
		}
	offset <<= 1;
	}

	if (tx == 0) 
	{
	if (gpu_sum != NULL)
	gpu_sum[blockIdx.x] = smem[n-1];
	smem[n-1] = 0;
	}
	
	for (int d = 1; d < n; d <<= 1)
	{
	offset >>= 1;
	__syncthreads();
		if (tx < d)
		{
		int ai = offset * (2*tx+1) - 1;
		int bi = offset * (2*tx+2) - 1;
		unsigned int t = smem[ai];
		smem[ai] = smem[bi];
		smem[bi] += t;
		}
	}
__syncthreads();
gpu_out[2*px] = smem[2*tx];
gpu_out[2*px+1] = smem[2*tx+1];
}

__global__ void increment( unsigned int * const d_array,const unsigned int * const d_incr)
{
	const size_t bx = 2 * blockDim.x * blockIdx.x;
	const size_t tx = threadIdx.x;
	const unsigned int u = d_incr[blockIdx.x];
	d_array[bx + 2*tx] += u;
	d_array[bx + 2*tx+1] += u;
}
void psum(const unsigned int * const h_in,unsigned int * const h_out,const size_t len)
{
unsigned int *d_in;
checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int)*len));

checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned int)*len, cudaMemcpyHostToDevice));
const unsigned int nthreads = MAX_THREADS_PER_BLOCK;
const unsigned int block_size = 2 * nthreads;
const unsigned int smem = block_size * sizeof(unsigned int);
const size_t n = len % block_size == 0 ? len : (1+len/block_size)*block_size;

int nblocks = n/block_size;

unsigned int *d_scan, *d_sums, *d_incr;
checkCudaErrors(cudaMalloc(&d_scan, sizeof(unsigned int)*n));
checkCudaErrors(cudaMalloc(&d_sums, sizeof(unsigned int)*nblocks));
checkCudaErrors(cudaMalloc(&d_incr, sizeof(unsigned int)*nblocks));

block_sum<<<nblocks, nthreads, smem>>>(d_in, d_scan, d_sums, block_size);
cudaDeviceSynchronize(); 
checkCudaErrors(cudaGetLastError());

block_sum<<<1, nthreads, smem>>>(d_sums, d_incr, NULL, block_size);
cudaDeviceSynchronize();
checkCudaErrors(cudaGetLastError());

increment<<<nblocks, nthreads>>>(d_scan, d_incr);
cudaDeviceSynchronize(); 
checkCudaErrors(cudaGetLastError());

checkCudaErrors(cudaMemcpy(h_out, d_scan, sizeof(unsigned int)*len, cudaMemcpyDeviceToHost));

checkCudaErrors(cudaFree(d_incr));
checkCudaErrors(cudaFree(d_sums));
checkCudaErrors(cudaFree(d_scan));
}

int main(int argc, char *argv[])
{
const size_t len = 1000;
thrust::host_vector<unsigned int> h_in(len);
thrust::host_vector<unsigned int> h_out(len);
	for (size_t i = 0; i < h_in.size(); i++)
	h_in[i] = 3*i;
	psum(&h_in[0], &h_out[0], len);
		for (size_t i = 0; i < h_in.size(); i++)
		std::cout << h_in[i] << " " << h_out[i] << std::endl;
		return 0;
}