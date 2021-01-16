#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N_size 16 		//number of elements in array 
#define thread_number 4   		//number of threads per block
#define block_number 4		//number of blocks 


__global__ void prescan(float *gpu_outdata, float *gpu_indata, int n);
void scanCPU(float *f_out, float *f_in, int i_n);

double myDiffTime(struct timeval &start, struct timeval &end)
{
double d_start, d_end;
d_start = (double)(start.tv_sec + start.tv_usec/1000000.0);
d_end = (double)(end.tv_sec + end.tv_usec/1000000.0);
return (d_end - d_start);
}

__global__ void prescan(float *gpu_outdata, float *gpu_indata, int n)
{
	extern  __shared__  float temp[];
	int thid = threadIdx.x;
	int bid = blockIdx.x;
	int offset = 1;
		if(bid * thread_number + thid<n)
		{ 
			temp[bid * thread_number + thid]  = gpu_indata[bid * thread_number + thid];
		}
		else
		{ 
			temp[bid * thread_number + thid]  = 0;
		} 
	
	for (int d = thread_number>>1; d > 0; d >>= 1)
	{
    		__syncthreads();
    		if (thid < d)
    		{
        	int ai = bid * thread_number + offset*(2*thid+1)-1;
        	int bi = bid * thread_number + offset*(2*thid+2)-1;
        	temp[bi] += temp[ai];
    		}
    	offset *= 2;
	}

	if (thid == 0)
	{
	    temp[thread_number - 1] = 0;
	}


	for (int d = 1; d < thread_number; d *= 2)
	{
	   offset >>= 1;
    	__syncthreads();
    		if (thid < d)
    		{
        	int ai = bid * thread_number + offset*(2*thid+1)-1;
        	int bi = bid * thread_number + offset*(2*thid+2)-1;
        	float t = temp[bid * thread_number + ai];
        	temp[ai]  = temp[ bi];
        	temp[bi] += t;
    		}
	}
	__syncthreads();

	gpu_outdata[bid * thread_number + thid] = temp[bid * thread_number + thid];
}

void scanCPU(float *f_out, float *f_in, int i_n)
{
	f_out[0] = 0;
	for (int i =1; i <=i_n; i++)
	{
    	f_out[i] = f_out[i-1] + f_in[i-1];
    	}
}

int main()
{
float a[N_size]={2.0,1.0,3.0,1.0,0.0,4.0,1.0,2.0,0.0,3.0,1.0,2.0,5.0,3.0,1.0,2.0}, c[N_size], g[N_size];
timeval start, end;

float *dev_a, *dev_g;
int size = N_size * sizeof(float);

double d_gpuTime, d_cpuTime;

	for (int i = 0; i < N_size; i++)
	{
    	printf("each element of an array a[%i] = %f\n", i, a[i]);
	}

cudaMalloc((void **) &dev_a, size);
cudaMalloc((void **) &dev_g, size);

gettimeofday(&start, NULL);

cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);

cudaDeviceSynchronize();

cudaMemcpy(g, dev_g, size, cudaMemcpyDeviceToHost);

gettimeofday(&end, NULL);
d_gpuTime = myDiffTime(start, end);

gettimeofday(&start, NULL);
scanCPU(c, a, N_size);

gettimeofday(&end, NULL);
d_cpuTime = myDiffTime(start, end);

cudaFree(dev_a); 
cudaFree(dev_g);

	for (int i = 0; i <=N_size; i++)
	{
    		printf("c[%i] = %0.3f, g[%i] = %0.3f\n", i, c[i], i, g[i]);
	}

printf("GPU Time for array size %i: %f\n", N_size, d_gpuTime);
printf("CPU Time for array size %i: %f\n", N_size, d_cpuTime);

}


