#include <cuda.h>

__global__ void InitDelaysKernel(unsigned int *delays, unsigned int nchans, float dm, float ftop, float foff, float tsamp);

__global__ void TransposeKernel(unsigned char *indata, unsigned char *outdata, unsigned int insize, unsigned int perthread);

__global__ void DedisperseKernel(unsigned char *indata, float *outdata, unsigned int *delays, unsigned int insamples, unsigned int outsamples, unsigned int perblock, unsigned int perthread);
