#ifndef _H_NINJECTOR_FILTERBANK
#define _H_NINJECTOR_FILTERBANK
#include <fstream>
#include <string>

struct FilHeader
{
	std::string rawfile;		// raw data file name
	std::string sourcename;	// source name
	int machineid;			// machine id
	int telescopeid;			// telescope id
	double ra;			// source right ascension
	double dec;			// source declinatio
	double az;			// azimuth angle in deg
	double zn;			// zenith angle in deg
	int datatype;			// data type ID
	double rdm;			// reference DM
	int nchans;			// number of channels
	double topfreq;			// frequency of the top channel MHz
	double chanband;			// channel bandwidth in MHz
	int nbeams;			// number of beams
	int ibeam;			// beam number
	int nbits;			// bits per sample
	double tstart;			// observation start time in MJD format
	double tsamp;			// sampling time in seconds
	int nifs;			// something
	size_t nsamps;			// number of time samples per channel
};

class Filterbank {
	private:
		unsigned char *fildata;	// filterbank data

		double topfreq;			// frequency of the top channel in MHz
		double fullband;		// full bandwidth in MHz

		FilHeader header;

		float obstime;			// observation stat time in MJD
		float samptime;			// sampling time in s

		unsigned int nbits;		// number of bits per sample
		unsigned int nchans;	// number of channels

	protected:

	public:
		Filterbank(std::string infilestr);
		~Filterbank(void);
		FilHeader GetHeader(void) const {return header;}
		bool ReadHeader(std::ifstream);
		void SaveFilterbank(std::string outfilestr) const;
		unsigned char* GetFilData(void) {return this -> fildata;}

};

#endif
