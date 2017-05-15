#include <cuda.h>

__global__ void InitDelaysKernel(unsigned int *delays, unsigned int nchans, float dm, float ftop, float foff, float tsamp);
