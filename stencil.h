#ifndef __STENCIL_H__
#define __STENCIL_H__

void serial_stencil(int width, int height, int time, float min, float max,
		    float *initial_state);

void threaded_stencil(int width, int height, int time, float min, float max, int threads,
		      float *initial_state);
void *threaded_update(void *args);

void openmp_stencil(int width, int height, int time, float min, float max, int threads,
		    float *initial_state);

#endif
