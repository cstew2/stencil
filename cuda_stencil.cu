#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_stencil.cuh"
#include "shared.cuh"
#include "render.h"

const int BLOCK_SIZE = 32;

void cuda_stencil(int width, int height, int time, float min, float max,
		      float *initial_state)
{
	int n = width*height;
	float *a;
	cudaMalloc((void **) &a, n * sizeof(float));
	float *b;
	cudaMalloc((void **) &b, n * sizeof(float));
	float *temp;

	cudaMemcpy(a, initial_state, n * sizeof(float), cudaMemcpyHostToDevice);

        uint32_t *image;
        cudaMalloc((void **) &image, n * sizeof(uint32_t));

	int block_size = BLOCK_SIZE;
	int grid_size = (n + block_size -1) / block_size;
	
	for(int ticks=0; ticks < time; ticks++) {
		
		//perform compuation
		cuda_update<<<grid_size, block_size>>>(a, b, n);

		//render image
		to_colour(a, min, max, n, image);
		gl_render(image, width, height);
		if(gl_update()) {
			break;
		}

		//swap state buffers
		temp = a;
		a = b;
		b = temp;
		temp = NULL;
	}
	
	cudaFree(image);
	cudaFree(a);
	cudaFree(b);	
}

__global__ void cuda_update(float *a, float *b, int n)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int sx = blockDim.x;
	int sy = blockDim.y;		
		
	int aBegin = n * blockDim.x * by;
	int aEnd = aBegin + n - 1;
	int aStep = blockDim.x;

	int bBegin = blockDim.x * bx;
	int bStep = blockDim.x * n;

	for(int i = aBegin, j = bBegin;  i <= aEnd; i += aStep, j += bStep) {
		//load block into shared memory
		__shared__ float s_cache[BLOCK_SIZE*BLOCK_SIZE];
		__shared__ float s_result[BLOCK_SIZE*BLOCK_SIZE];
		s_cache[ty * sx + tx] = a[i + n * ty + tx];
		__syncthreads();

		//find neighbours then perform operation
		float neighbours[9] = {0};
		von_neumann_neighbours(tx, ty, sx, sy, s_cache, neighbours);
		s_result[ty * sx + tx] = average(9, neighbours);
		__syncthreads();

		//writeback
		b[(n * blockDim.x * by + blockDim.x * bx) + n * ty + tx] = s_result[ty * sx + tx];
	}	
}


