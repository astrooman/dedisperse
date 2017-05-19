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

    bool verbose = false;

    double dm = 0;

    string infile;

    unsigned int outbands = 0;

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
            } else if (string(argv[iarg]) == "-v") {
                verbose = true;
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

    cout << "Read the file..." << endl;

    std::ofstream origfile("original.ascii");

/*
    if (!origfile)
        cerr << "Could not create the original output file!" << endl;

    for (int isamp = 0; isamp < filhead.nsamps; isamp++) {
        for (int ichan = 0; ichan < filhead.nchans; ichan++) {
            origfile << (int)*(filfile.GetFilData() + isamp * filhead.nchans + ichan) << " ";
        }
        origfile << endl;
    }

    cout << "Saved the original file..." << endl;
*/
    origfile.close();
    // NOTE: Assume the whole file will fit in the device memory at once for now
    size_t insamps = filhead.nsamps;

    if (verbose)
        cout << "Number of input samples: " << insamps << endl;

    size_t insize = insamps * filhead.nchans;      // total number of points in the input filterbank file
    unsigned char *indata;
    cudaCheckError(cudaMalloc((void**)&indata, insize * sizeof(unsigned char)));
    cudaCheckError(cudaMemcpy(indata, filfile.GetFilData(), insize * sizeof(unsigned char), cudaMemcpyHostToDevice));
/*
    unsigned char *indata2 = new unsigned char[insize];
    cudaCheckError(cudaMemcpy(indata2, indata, insize * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    std::ofstream origfile2("original2.ascii");

    if (!origfile2)
        cerr << "Could not create the original output file!" << endl;

    for (int isamp = 0; isamp < filhead.nsamps; isamp++) {
        for (int ichan = 0; ichan < filhead.nchans; ichan++) {
            origfile2 << (int)*(filfile.GetFilData() + isamp * filhead.nchans + ichan) << " ";
        }
        origfile2 << endl;
    }

    cout << "Saved the second original file..." << endl;

    origfile2.close();
*/
    unsigned int *ddelays;
    cudaCheckError(cudaMalloc((void**)&ddelays, filhead.nchans * sizeof(unsigned int)));
    dim3 nthreads(min(filhead.nchans, 1024), 1, 1);
    dim3 nblocks(nthreads.x / filhead.nchans, 1, 1);
    cout << "Generating the delays..." << endl;
    InitDelaysKernel<<<nblocks, nthreads, 0>>>(ddelays, filhead.nchans, dm, filhead.topfreq, filhead.chanband, filhead.tsamp);
    cudaCheckError(cudaDeviceSynchronize());
    cout << "Delays generated..." << endl;
    unsigned int *hdelays = new unsigned int[filhead.nchans];
    cudaCheckError(cudaMemcpy(hdelays, ddelays, filhead.nchans * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    cout << "Generated delays: " << endl;
    for (int ichan = 0; ichan < filhead.nchans; ichan++)
        cout << "\tChannel " << ichan << ": " << hdelays[ichan] << endl;

    unsigned char *transdata;
    cudaCheckError(cudaMalloc((void**)&transdata, insize * sizeof(unsigned char)));

    unsigned int perblock = 16;

    nthreads.x = min(filhead.nchans, 1024);
    nblocks.x =  (insamps - 1) / perblock + 1;
    unsigned int perthread = filhead.nchans / nthreads.x;

    cout << "Transposing the data..." << endl;
    cout << perthread << " channels per thread..." << endl;
    cout << nblocks.x << " blocks in x dimension..." << endl;
    TransposeKernel<<<nblocks, nthreads, 0>>>(indata, transdata, insamps, perthread, perblock);
    cudaCheckError(cudaDeviceSynchronize());
    cout << "Data has been transposed..." << endl;

    unsigned char *trans = new unsigned char[insize];
    cudaCheckError(cudaMemcpy(trans, transdata, insize * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    std::ofstream transfile("trans.ascii");

    if (!transfile) {
        cerr << "Could not create the output file!" << endl;
        exit(EXIT_FAILURE);
    }

    for (int ichan = 0; ichan < filhead.nchans; ichan++) {
        for (int isamp = 0; isamp < insamps; isamp++) {
            transfile << (int)trans[ichan * filhead.nsamps + isamp] << " ";
        }
        transfile << endl;
    }

    transfile.close();

    delete [] trans;

    unsigned int maxdelay = hdelays[filhead.nchans - 1];
    size_t outsamps = filhead.nsamps - maxdelay;
    if (verbose)
        cout << "Number of output samples: " << outsamps << endl;
    size_t outsize = outbands * outsamps;
    float *outdata;
    cudaCheckError(cudaMalloc((void**)&outdata, outsize * sizeof(float)));

    perblock = 4;
    nblocks.x = ((outsamps - 1) / perblock) + 1;
    if (verbose) {
        cout << "Dedispersing the data..." << endl;
        cout << "Number of blocks to use: " << nblocks.x << endl;
        cout << "Threads per block: " << nthreads.x << endl;
    }

    // DedisperseKernel<<<nblocks, nthreads, 0>>>(transdata, outdata, ddelays, filhead.nsamps, outsamps, perblock, perthread);
    nthreads.x = outbands;
    perthread = filhead.nchans / nthreads.x;
    DedisperseBandKernel<<<nblocks, nthreads, 0>>>(transdata, outdata, ddelays, filhead.nsamps, outsamps, perblock, perthread, outbands);
    cudaCheckError(cudaDeviceSynchronize());
    cout << "Data has been dedispersed..." << endl;

    float *saved = new float[outsize];
    cudaCheckError(cudaMemcpy(saved, outdata, outsize * sizeof(float), cudaMemcpyDeviceToHost));

    std::ofstream outfile("dedisp.ascii");

    if (!outfile) {
        cerr << "Could not create the output file!" << endl;
        exit(EXIT_FAILURE);
    }

    cout << "Saving the ASCII data..." << endl;

    for (unsigned int ichan = 0; ichan < outbands; ichan++) {
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
