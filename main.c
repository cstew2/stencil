#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#include "render.h"
#include "stencil.h"
#include "cuda_stencil.cuh"

int main(int argc, char **argv)
{
	bool cuda = false;
	bool posix = false;
	bool openmp = false;
	int threads = 0;
	for(int i=1; i < argc; i++) {
		if(!strcmp(argv[i], "--cuda")) {
			cuda = true;
		}
		else if(!strcmp(argv[i], "--thread")) {
			posix = true;
			threads = atoi(argv[++i]);
		}
		else if(!strcmp(argv[i], "--openmp")) {
			openmp = true;
		}
       	}
	
	int width = 500;
	int height = 500;
	int time = 500000;
	float min = 0;
	float max = 10000;
	float *initial_state = (float *) calloc(width*height, sizeof(float)); 

	int bar_edge = width/32;
	int bar_size = width/8;

	
	for(int y=1; y < height-1; y++) {
		for(int x=1; x < width-1; x++) {
			if((x > bar_edge                  && x < bar_size+bar_edge) ||
			   (x > width-(bar_size+bar_edge) && x < width - bar_edge)) {
				initial_state[y * width + x] = max;
			}
		}
	}
	
	if(gl_init(width, height)) {
		return -1;
	}

	if(cuda) {
		printf("Using CUDA\n");
		cuda_stencil(width, height, time, min, max,
				 initial_state);
	}
	else if(posix) {
		printf("Using posix threads\n");
		threaded_stencil(width, height, time, min, max, threads,
				 initial_state);
	}
	else if(openmp) {
		printf("Using openmp\n");
		openmp_stencil(width, height, time, min, max, threads,
				 initial_state);
	}
	else {
		printf("Serial\n");
		serial_stencil(width, height, time, min, max,
			       initial_state);
	}

	gl_cleanup();
	
	free(initial_state);
}
