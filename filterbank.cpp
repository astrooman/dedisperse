#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <filterbank.hpp>	// Filterbank file declaration, header_s structure

using std::cerr;
using std::cout;
using std::endl;
using std::ios;
using std::string;

Filterbank::Filterbank(string infilestr) {
	cout << "Reading the file " << infilestr << "...\n";
	cout << "Reading the header...\n";

	std::ifstream infile(infilestr.c_str(), std::ifstream::in | std::ifstream::binary);

	if(!infile) {
		cerr << "Could not open the input file " << infilestr << endl;
		cerr << "Will now quit!\n";
		exit(EXIT_FAILURE);
	}

	string read_param;
	char field[60];

	int fieldlength;

	while(true)		// go 4eva
	{
		infile.read((char *)&fieldlength, sizeof(int));
		infile.read(field, fieldlength * sizeof(char));
		field[fieldlength] = '\0';
		read_param = field;

		if (read_param == "HEADER_END") break;		// finish reading the header when its end is reached
		else if (read_param == "rawdatafile") {
			infile.read((char *)&fieldlength, sizeof(int));		// reads the length of the raw data file name
			infile.read(field, fieldlength * sizeof(char));
			field[fieldlength] = '\0';
			header.rawfile = field;
		}
		else if (read_param == "source_name") {
			infile.read((char *)&fieldlength, sizeof(int));
			infile.read(field, fieldlength * sizeof(char));
			field[fieldlength] = '\0';
			header.sourcename = field;
		}
		else if (read_param == "machine_id")	infile.read((char *)&header.machineid, sizeof(int));
		else if (read_param == "telescope_id")	infile.read((char *)&header.telescopeid, sizeof(int));
		else if (read_param == "src_raj")	infile.read((char *)&header.ra, sizeof(double));
		else if (read_param == "src_dej")	infile.read((char *)&header.dec, sizeof(double));
		else if (read_param == "az_start")	infile.read((char *)&header.az, sizeof(double));
		else if (read_param == "za_start")	infile.read((char *)&header.zn, sizeof(double));
		else if (read_param == "data_type")	infile.read((char *)&header.datatype, sizeof(int));
		else if (read_param == "refdm")		infile.read((char *)&header.rdm, sizeof(double));
		else if (read_param == "nchans")	infile.read((char *)&header.nchans, sizeof(int));
		else if (read_param == "fch1")		infile.read((char *)&header.topfreq, sizeof(double));
		else if (read_param == "foff")		infile.read((char *)&header.chanband, sizeof(double));
		else if (read_param == "nbeams")	infile.read((char *)&header.nbeams, sizeof(int));
		else if (read_param == "ibeam")		infile.read((char *)&header.ibeam, sizeof(int));
		else if (read_param == "nbits")		infile.read((char *)&header.nbits, sizeof(int));
		else if (read_param == "tstart")	infile.read((char *)&header.tstart, sizeof(double));
		else if (read_param == "tsamp")		infile.read((char *)&header.tsamp, sizeof(double));
		else if (read_param == "nifs")		infile.read((char *)&header.nifs, sizeof(int));
	}

	size_t headerendpos = infile.tellg();
	infile.seekg(0, infile.end);
	size_t fileendpos = infile.tellg();
	infile.seekg(headerendpos, infile.beg);
	header.nsamps = (fileendpos - headerendpos) / header.nchans * (8 / header.nbits);

	size_t toread = header.nsamps * header.nchans * header.nbits / 8;
	this->fildata = new unsigned char[toread];
	infile.read(reinterpret_cast<char*>(this->fildata), toread);
	infile.close();

}

void Filterbank::SaveFilterbank(string outfilestr) const {
	std::ofstream outfile(outfilestr.c_str(), std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);

	if(!outfile) {
		cerr << "Could not create the output file " << outfile << endl;
		cerr << "Will now quit!\n";
		exit(EXIT_FAILURE);
	}

	cout << "Saving the file...\n";

	int strlen;
	char field[60];

	strlen = 12;
	// header start - MUST be at the start!!
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "HEADER_START");
	outfile.write(field, strlen * sizeof(char));

	//telescope id
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "telescope_id");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.telescopeid, sizeof(int));

	strlen = 11;
	// raw data file name
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "rawdatafile");
	outfile.write(field, strlen * sizeof(char));
	// need to restart after that
	strlen = header.rawfile.size();
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, header.rawfile.c_str());
	outfile.write(field, strlen * sizeof(char));

	strlen = 11;
	//source name
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "source_name");
	outfile.write(field, strlen * sizeof(char));
	// need to restart after that
	strlen = header.sourcename.size();
	strcpy(field, header.sourcename.c_str());
	outfile.write((char*)&strlen, sizeof(int));
	outfile.write(field, strlen * sizeof(char));

	strlen = 10;
	// machine id
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "machine_id");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.machineid, sizeof(int));

	strlen = 9;
	//data type
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "data_type");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.datatype, sizeof(int));

	strlen = 8;
	// azimuth
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "az_start");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.az, sizeof(double));

	// zenith
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "za_start");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.zn, sizeof(double));

	strlen = 7;
	// source right ascension
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "src_raj");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.ra, sizeof(double));

	// source declination
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "src_dej");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.dec, sizeof(double));

	strlen = 6;
	// first sample time stamp
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "tstart");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.tstart, sizeof(double));

	// number of filterbank channels
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "nchans");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.nchans, sizeof(int));

	// number of beams
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "nbeams");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.nbeams, sizeof(int));

	strlen = 5;
	// sampling interval
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "tsamp");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.tsamp, sizeof(double));

	// bits per time sample
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "nbits");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.nbits, sizeof(int));

	// reference dispersion measure
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "refdm");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.rdm, sizeof(double));

	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "ibeam");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.ibeam, sizeof(int));

	strlen = 4;
	// top channel frequency
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "fch1");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.topfreq, sizeof(double));

	// channel bandwidth
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "foff");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.chanband, sizeof(double));

	// number of if channels
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "nifs");
	outfile.write(field, strlen * sizeof(char));
	outfile.write((char*)&header.nifs, sizeof(int));

	strlen = 10;
	// header end - MUST be at the end!!
	outfile.write((char*)&strlen, sizeof(int));
	strcpy(field, "HEADER_END");
	outfile.write(field, strlen *sizeof(char));

	cout << "Header saved!\n";
	cout << "Saving the data now...\n";
	size_t tosave = header.nsamps * header.nchans * header.nbits / 8;
	cout << "Will save " << (double)tosave / 1024.0 / 1024.0 / 1024.0 << "GB of data\n";
	outfile.write(reinterpret_cast<char*>(fildata), tosave);
	outfile.close();
}

Filterbank::~Filterbank() {
	delete [] this -> fildata;
}
