#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "errors.hpp"
#include "filterbank.hpp"
#include "kernels.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

int main(int argc, char *argv[]) {

    double dm{0};

    string infile;

    unsigned int outbands{0};

    if (argc >= 2) {
        for (int iarg = 0; iarg < argc; iarg  ++) {
            if (string(argv[iarg]) == "-f") {
                iarg++;
                infile = string(argv[iarg]);
            } else if (string(argv[iarg]) == "-d") {
                iarg++;
                dm = atof(argv[iarg]);
            } else if (string(argv[iarg]) == "-b") {
                iarg++;
                outbands = atoi(argv[iarg]);
            }

	    }
    }

    Filterbank filfile(infile);
    FilHeader filhead = filfile.GetHeader();
    // NOTE: Don't perform any frequency scrunching if user didn't set the number of output bands
    if (outbands == 0) {
        outbands = filhead.nchans;
    }
    // TODO: Set outbands to the nearest power of 2 for my own sake

    // NOTE: Assume the whole file will fit in the device memory at once for now
    size_t insamps = filhead.nsamps;

    size_t insize = insamps * filhead.nchans;      // total number of points in the input filterbank file
    unsigned char *indata;
    cudaCheckError(cudaMalloc((void**)&indata, insize * sizeof(unsigned char)));

    unsigned int *ddelays;
    cudaCheckError(cudaMalloc((void**)&ddelays, filhead.nchans * sizeof(unsigned int)));
    dim3 nthreads(min(filhead.nchans, 1024), 1, 1);
    dim3 nblocks(nthreads.x / filhead.nchans, 1, 1);
    InitDelaysKernel<<<nblocks, nthreads, 0>>>(ddelays, filhead.nchans, dm, filhead.topfreq, filhead.chanband, filhead.tsamp);
    cudaCheckError(cudaDeviceSynchronize());

    unsigned int *hdelays = new unsigned int[filhead.nchans];
    cudaCheckError(cudaMemcpy(hdelays, ddelays, filhead.nchans * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    unsigned char *transdata;
    cudaCheckError(cudaMalloc((void**)&transdata, filhead.nchans * filhead.nsamps * sizeof(unsigned char)));
    nthreads.x = min(filhead.nchans, 1024);
    nblocks.x = insamps;
    unsigned int perthread = filchans.nchans / nthreads.x;
    TransposeKernel<<<nblocks, nthreads, 0>>>(indata, transdata, perthread)
    cudaCheckError(cudaDeviceSynchronize());

    unsigned int maxdelay = hdelays[filhead.nchans - 1];
    size_t outsamps = filhead.nsamps - maxdelay
    size_t outsize = filhead.nchans * outsamps;
    float *outdata;
    cudaCheckError(cudaMalloc((void**)&outdata, outsize * sizeof(float)));

    perblock = 4;
    nblocks.x = (outsamps - 1 / perblock) + 1;
    DedisperseKernel<<<nblocks, nthreads, 0>>>(transdata, outdata, ddelays, filhead.nsamps, outsamps, perblock, perthread);
    cudaCheckError(cudaDeviceSynchronize());

    float *saved = new float[outsize];
    cudaCheckError(cudaMemcpy(saved, outdata, outsize * sizeof(float)));

    std::ofstream outfile("dedisp.ascii");

    if !(outfile) {
        cerr << "Could not create the output file!" << endl;
        exit(EXIT_FAILURE);
    }

    for (unsigned int ichan = 0; ichan < outbands; ichan++ {
        for (unsigned int isamp = 0; isamp < outsamps; isamp++) {
            outfile << saved[ichan * outsamps + isamp] << " ";
        }
        outfile << endl;
    }

    outfile.close();

    delete [] saved;
    delete [] hdelays;

    cudaCheckError(cudaFree(outdata));
    cudaCheckError(cudaFree(transdata));
    cudaCheckError(cudaFree(ddelays));
    cudaCheckError(cudaFree(indata));

    return 0;
}
