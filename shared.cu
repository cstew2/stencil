#include "shared.cuh"

__device__ __host__ void moore_neighbours(int x, int y, int width, int height,
					  float *state, float neighbours[8])
{
	int c[8][2] = {{1,0}, {0,1}, {1,1}, {-1,0}, {0,-1}, {1,-1}, {-1,1}, {-1,-1}};
	for(int i=0; i < 8; i++) {
		neighbours[i] = state[index_1d(x+c[i][0], y+c[i][1], width, height)];
	}
}
__device__ __host__ void von_neumann_neighbours(int x, int y, int width, int height,
						float *state, float neighbours[4])
{
	int c[4][2] = {{1,0}, {0,1}, {-1,0}, {0,-1}};
	for(int i=0; i < 4; i++) {
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

void to_colour(float *state, float min, float max, int count, uint32_t *colours)
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
