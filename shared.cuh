#ifndef __SHARED_CUH__
#define __SHARED_CUH__

#include <stdint.h>

#ifdef __CUDACC__
__device__ __host__
#endif
void moore_neighbours(int x, int y, int width, int height,
					  float *state, float neighbours[9]);

#ifdef __CUDACC__		      
__device__ __host__
#endif
void von_neumann_neighbours(int x, int y, int width, int height,
			    float *state, float neighbours[5]);
#ifdef __CUDACC__
__device__ __host__
#endif
float average(int count, float* neighbours);

#ifdef __CUDACC__
__device__ __host__
#endif
float diff_average(int count, float value, float* neighbours);

#ifdef __CUDACC__
__device__ __host__
#endif
int index_1d(int x, int y, int width, int height);

#ifdef __CUDACC__
__device__ __host__
#endif
 int index_1d_mod(int x, int y, int width, int height);

void to_colour(float *state, float min, float max, int count, uint32_t *colours);

#endif
