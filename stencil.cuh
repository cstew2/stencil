#ifndef __STENCIL_CUH__
#define __STENCIL_CUH__

void serial_stencil(int width, int height, int time, float min, float max,
		    float *initial_state);

__device__ __host__ void moore_neighbours(int x, int y, int width, int height,
					  float *state, float neighbours[9]);
__device__ __host__ void von_neumann_neighbours(int x, int y, int width, int height,
						float *state, float neighbours[5]);

__device__ __host__ float average(int count, float* neighbours);
__device__ __host__ float diff_average(int count, float value, float* neighbours);

__device__ __host__ void to_colour(float *state, float min, float max, int count, uint32_t *colours);

__device__ __host__ int index_1d(int x, int y, int width, int height);
__device__ __host__ int index_1d_mod(int x, int y, int width, int height);

#endif
