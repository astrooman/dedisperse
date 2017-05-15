#include <cuda.h>

#include "kernels.cuh"

__global__ void InitDelaysKernel(unsigned int *delays, unsigned int nchans, float dm, float ftop, float foff, float tsamp) {
    tsamp *= 1e+03;     // conver sampling time from seconds to milliseconds
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float fbott = ftop + (float)idx * foff;
    // NOTE: The below calculation is done for delay in milliseconds with frequencies in MHz
    delays[idx] = (int)(4.15e+06 * ( 1.0f / (fbott * fbott) - 1.0f / (ftop * ftop)) * dm / tsamp);
}
