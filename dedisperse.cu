#include <cstdlib>
#include <iostream>
#include <string>

#include "errors.hpp"
#include "filterbank.hpp"
#include "kernels.cuh"

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
    size_t insize = filhead.nsamps * filhead.nchans;      // total number of points in the input filterbank file
    unsigned char *indata;
    cudaCheckError(cudaMalloc((void**)&indata, insize * sizeof(unsigned char)));

    unsigned int *delays;
    cudaCheckError(cudaMalloc((void**)&delays, filhead.nchans * sizeof(unsigned int)));
    dim3 nthreads(min(filhead.nchans, 1024), 1, 1);
    dim3 nblocks(nthreads.x / filhead.nchans, 1, 1);
    InitDelaysKernel<<<nblocks, nthreads, 0>>>(delays, filhead.nchans, dm, filhead.topfreq, filhead.chanband, filhead.tsamp);


    return 0;
}
