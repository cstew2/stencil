#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#include <unistd.h>
#include <pthread.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "stencil.cuh"
#include "render.h"

int delay = 0;

struct thread_args {
	int x;
	int y;
	int width;
	int height;
	float *a;
	float *b;
};

int main(int argc, char **argv)
{
	bool cuda = false;
	int threads = 0;
	for(int i=1; i < argc; i++) {
		if(strcmp(argv[i], "--cuda")) {
			cuda = true;
		}
		else if(strcmp(argv[i], "--thread")) {
			threads = atoi(argv[++i]);
		}
       	}
	
	int width = 2500;
	int height = 2500;
	int time = 500000;
	float min = 0;
	float max = 10000;
	float *initial_state = (float *) calloc(width*height, sizeof(float)); 

	int bar_edge = width/32;
	int bar_size = width/16;
	
	for(int y=1; y < height-1; y++) {
		for(int x=1; x < width-1; x++) {
			if((x > bar_edge                  && x < bar_size+bar_edge) ||
			   (x > width-(bar_size+bar_edge) && x < width - bar_edge)) {
				initial_state[index_1d(x, y, width, height)] = max;
			}
		}
	}
	
	gl_init(width, height);

	if(cuda) {
		
		parallel_stencil(width, height, time, min, max,
				 initial_state);
	}
	else if(threads > 0) {
		threaded_stencil(width, height, time, min, max, threads,
				 initial_state);
	}
	else {
		serial_stencil(width, height, time, min, max,
			       initial_state);
	}

	gl_cleanup();
	
	free(initial_state);
}

void serial_stencil(int width, int height, int time, float min, float max,
		    float *initial_state)
{
	int num = width*height;
	float *a = (float *) calloc(num, sizeof(float));
	float *b = (float *) calloc(num, sizeof(float));
	float *temp;

	memcpy(a, initial_state, num * sizeof(float));

	uint32_t *image = (uint32_t *) calloc(num, sizeof(uint32_t));
	
	int n_count = 5;
	float neighbours[n_count] = {0};
	
	for(int ticks=0; ticks < time; ticks++) {
		for(int y=1; y < height-1; y++) {
			for(int x=1; x < width-1; x++) {
				von_neumann_neighbours(x, y, width, height, a, neighbours);
				b[index_1d(x, y, width, height)] = average(n_count, neighbours);
			}
		}
	
		//render image
		to_colour(a, min, max, num, image);
		gl_render(image, width, height);
		if(gl_update()) {
			break;
		}
			
		//swap state buffers
		temp = a;
		a = b;
		b = temp;
		temp = NULL;
		//usleep(delay);
	}
	
	free(image);
	free(a);
	free(b);
}

void threaded_stencil(int width, int height, int time, float min, float max, int threads,
		      float *initial_state)
{	
	int num = width*height;
	float *a = (float *) calloc(num, sizeof(float));
	float *b = (float *) calloc(num, sizeof(float));
	float *temp;

	memcpy(a, initial_state, num * sizeof(float));

	uint32_t *image = (uint32_t *) calloc(num, sizeof(uint32_t));

	pthread_t thread[threads];
	pthread_attr_t attr;
	struct thread_args args;
	
        pthread_attr_init(&attr);
	int e = 0;
	
	for(int ticks=0; ticks < time; ticks++) {
		
		for(int i=0; i < threads; i++) {		
			args = {1, 1, width, height, a, b};
			e = pthread_create(&thread[i], &attr, &threaded_update, &args);
			if(e != 0) {
				printf("pthread_create error: %d\n", e);
				return;
			}
		}

		for(int i=0; i < threads; i++) {
			pthread_join(thread[i], NULL);
		}
	
		//render image
		to_colour(a, min, max, num, image);
		gl_render(image, width, height);
		if(gl_update()) {
			break;
		}
		
		//swap state buffers
		temp = a;
		a = b;
		b = temp;
		temp = NULL;

		usleep(delay);
	}
	
	free(image);
	free(a);
	free(b);
}

void parallel_stencil(int width, int height, int time, float min, float max,
		      float *initial_state)
{
	int num = width*height;
	float *a;
	cudaMalloc((void **) &a, num * sizeof(float));
	float *b;
	cudaMalloc((void **) &b, num * sizeof(float));
	float *temp;

	cudaMemcpy(a, initial_state, num * sizeof(float), cudaMemcpyHostToDevice);

        uint32_t *image;
        cudaMalloc((void **) &image, num * sizeof(uint32_t));
		
	for(int ticks=0; ticks < time; ticks++) {
		//perform compuation
		//parallel_update<<<256, 256, 256>>>(a, b);
		
		//render image
		to_colour(a, min, max, num, image);
		gl_render(image, width, height);
		if(gl_update()) {
			break;
		}

		//swap state buffers
		temp = a;
		a = b;
		b = temp;
		temp = NULL;

		usleep(delay);
	}
	
	cudaFree(image);
	cudaFree(a);
	cudaFree(b);	
}

void *threaded_update(void *args)
{
	struct thread_args arg = *((thread_args *) args);
	
	int n_count = 9;
	float neighbours[n_count] = {0};
	for(int i=arg.y; i < arg.height; i++) {
		for(int j=arg.x; j < arg.width; j++) {
			moore_neighbours(i, j, arg.width, arg.height, arg.a, neighbours);
			arg.b[index_1d(i, j, arg.width, arg.height)] = average(n_count, neighbours);
		}
	}
	return 0;
}

__device__ void parallel_update(float *a, float *b)
{
	float neighbours[9] = {0};
	von_neumann_neighbours(threadIdx.x, threadIdx.y, blockDim.x, blockDim.y, a, neighbours);
	b[index_1d(threadIdx.x, threadIdx.y, threadIdx.y, blockDim.y)] = average(9, neighbours);
}

__device__ __host__ void moore_neighbours(int x, int y, int width, int height,
					  float *state, float neighbours[9])
{
	int c[9][2] = {{0,0}, {1,0}, {0,1}, {1,1}, {-1,0}, {0,-1}, {1,-1}, {-1,1}, {-1,-1}};
	for(int i=0; i < 9; i++) {
		neighbours[i] = state[index_1d(x+c[i][0], y+c[i][1], width, height)];
	}
}
__device__ __host__ void von_neumann_neighbours(int x, int y, int width, int height,
						float *state, float neighbours[5])
{
	int c[5][2] = {{0,0}, {1,0}, {0,1}, {-1,0}, {0,-1}};
	for(int i=0; i < 5; i++) {
		neighbours[i] = state[index_1d(x+c[i][0], y+c[i][1], width, height)];
	}
}

__device__ __host__ float average(int count, float* neighbours)
{
	float sum = 0;
	for(int i=0; i < count; i++) {
		sum += neighbours[i];
	}
	return sum/(float)count;
}

__device__ __host__ float diff_average(int count, float value, float* neighbours)
{
	float sum = 0;
	for(int i=0; i < count; i++) {
		sum += neighbours[i];
	}
	return value - (sum/(float)count);	
}

__device__ __host__ int index_1d(int x, int y, int width, int height)
{
	if(x < 0 || x > width || y < 0 || y > height) {
		return -1;
	}
	return (y * width) + x;
}

__device__ __host__ int index_1d_mod(int x, int y, int width, int height)
{
	int x_d = x % width;
	int y_d = y % height;
	return ((y_d < 0 ? y_d+height : y_d) * width) + (x_d < 0 ? x_d+width : x_d);
}

__host__ void to_colour(float *state, float min, float max, int count, uint32_t *colours)
{
	uint8_t R;
	uint8_t G;
	uint8_t B;
	uint8_t A = 0xFF;
	
	for(int i=0; i < count; i++) {
		//normalize to [0,1]
		float s_i = (state[i] - min)/(max - min);
		
		//calculate RGB value
		R =  (0xFF * s_i);
		G =   0x00;
		B = -(0xFF * s_i) + 0xFF;

		colours[i] = (A << 24) | (B << 16) | (G << 8) | R;
	}
}