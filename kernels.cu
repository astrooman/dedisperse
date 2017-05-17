#include <cuda.h>

#include "kernels.cuh"

__global__ void InitDelaysKernel(unsigned int *delays, unsigned int nchans, float dm, float ftop, float foff, float tsamp) {
    tsamp *= 1e+03;     // conver sampling time from seconds to milliseconds
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float fbott = ftop + (float)idx * foff;
    // NOTE: The below calculation is done for delay in milliseconds with frequencies in MHz
    delays[idx] = (int)(4.15e+06 * ( 1.0f / (fbott * fbott) - 1.0f / (ftop * ftop)) * dm / tsamp);
}

__global__ void TransposeKernel(unsigned char *indata, unsigned char *outdata, unsigned int insize, unsigned int perthread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < insize) {
        for (int ichan = 0; ichan < perthread; ichan++) {
            outdata[(threadIdx.x * perthread + ichan) * gridDim.x + blockIdx.x] = indata[idx * perthread + ichan];
        }
    }
}

__global__ void DedisperseKernel(unsigned char *indata, float *outdata, unsigned int *delays, unsigned int insamples, unsigned int outsamples, unsigned int perblock, unsigned int perthread) {

    unsigned int chanid;
    unsigned int sampid;

    for (unsigned int isamp = 0; isamp < perblock; isamp++) {
        sampid = blockIdx.x * perblock + isamp;
        if (sampid < outsamples) {
            for (unsigned int ichan = 0; ichan < perthread; ichan ++) {
                chanid = threadIdx.x * perthread + ichan;
                outdata[chanid * outsamples + sampid] = (float)indata[chanid * insamples + sampid + delays[chanid]];
            }
        }
    }

}
