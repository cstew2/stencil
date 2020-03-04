#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#include <unistd.h>
#include <pthread.h>

#include <omp.h>

#include "stencil.h"
#include "shared.cuh"
#include "render.h"

struct thread_args {
	int x;
	int y;
	int width;
	int height;
	float *a;
	float *b;
};

void serial_stencil(int width, int height, int time, float min, float max,
		    float *initial_state)
{
	int num = width*height;
	float *a = (float *) calloc(num, sizeof(float));
	float *b = (float *) calloc(num, sizeof(float));
	float *temp;

	memcpy(a, initial_state, num * sizeof(float));

	uint32_t *image = (uint32_t *) calloc(num, sizeof(uint32_t));
	
	int n_count = 4;
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
	}
	
	free(image);
	free(a);
	free(b);
}

void *threaded_update(void *args)
{
	struct thread_args arg = *((thread_args *) args);
	
	int n_count = 8;
	float neighbours[n_count] = {0};
	for(int i=arg.y; i < arg.height; i++) {
		for(int j=arg.x; j < arg.width; j++) {
			moore_neighbours(i, j, arg.width, arg.height, arg.a, neighbours);
			arg.b[index_1d(i, j, arg.width, arg.height)] = average(n_count, neighbours);
		}
	}
	return 0;
}

void openmp_stencil(int width, int height, int time, float min, float max, int threads,
		    float *initial_state)
{
	int num = width*height;
	float *a = (float *) calloc(num, sizeof(float));
	float *b = (float *) calloc(num, sizeof(float));
	float *temp;

	memcpy(a, initial_state, num * sizeof(float));

	uint32_t *image = (uint32_t *) calloc(num, sizeof(uint32_t));
	
	int n_count = 4;
	float neighbours[n_count] = {0};
	
	for(int ticks=0; ticks < time; ticks++) {
                #pragma omp parallel for private(neighbours) collapse(2)
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

	}
	
	free(image);
	free(a);
	free(b);	
}
