#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "stencil.cuh"

#include "render.h"

int main(void)
{
	int width = 100;
	int height = 100;
	int time = 500000;
	float min = 0;
	float max = 1000000;
	float *initial_state = (float *) calloc(width*height, sizeof(float)); 

	for(int y=1; y < height-1; y++) {
		for(int x=1; x < width-1; x++) {
			if(x == 2 || x == 3 || x == width-4 || x == width-3) {
				initial_state[(y * width) + x] = max;
			}
		}
	}

	gl_init(width, height);
	
	serial_stencil(width, height, time, min, max,
		       initial_state);

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
	
	int n_count = 9;
	float neighbours[n_count] = {0};
	
	for(int ticks=0; ticks < time; ticks++) {
		for(int y=1; y < height-1; y++) {
			for(int x=1; x < width-1; x++) {
				von_neumann_neighbours(x, y, width, height, a, neighbours);
				b[(y*width) + x] = average(n_count, neighbours);
			}
		}
	
		//render image
		to_colour(a, min, max, num, image);
		gl_render(image, width, height);

		//swap state buffers
		temp = a;
		a = b;
		b = temp;
		temp = NULL;

		usleep(100000);
	}

	free(a);
	free(b);
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

__device__ __host__ void to_colour(float *state, float min, float max, int count, uint32_t *colours)
{
	uint32_t R;
	uint32_t G;
	uint32_t B;
	uint32_t A = 0xFF000000;
	
	for(int i=0; i < count; i++) {
		//normalize to [0,1]
		float s_i = (state[i] - min)/(max - min);
		
		//calculate RGB value
		R = 0x000000FF * s_i + 0x00000000;
		G = 0x0000FF00 * 0;
		B = -0x00FF0000 * s_i + 0x00FF0000;

		colours[i] = R | G | B | A;
	}
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