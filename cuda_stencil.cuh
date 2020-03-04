#ifndef __STENCIL_CUH__
#define __STENCIL_CUH__

void cuda_stencil(int width, int height, int time, float min, float max,
		  float *initial_state);

#ifdef __CUDACC__
__global__
#endif
void cuda_update(float *a, float *b, int n);


#endif
