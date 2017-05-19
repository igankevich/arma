#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <string>

#include <unistd.h>

#include "config.hh"
#include "parallel_mt.hh"

void
generate_mersenne_twisters(std::ostream& out, size_t num_generators) {
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	arma::parallel_mt_seq<> seq(seed);
	std::ostream_iterator<arma::mt_config> out_it(out);
	size_t i = 0;
	std::generate_n(
		out_it,
		num_generators,
		[&i,&seq,num_generators] () {
			++i;
			std::clog << "Finished "
				<< '[' << i << '/' << num_generators << ']'
				<< std::endl;
			return seq();
		}
	);
}

void
usage(char* argv0) {
	std::cout
		<< "usage: "
		<< ARMA_DCMT_NAME
		<< " [-n <number>] [-o <path>] [-h]\n";
}

int
main(int argc, char* argv[]) {
	int ngenerators = 128;
	std::string filename = MT_CONFIG_FILE;
	bool help_requested = false;
	int opt = 0;
	while ((opt = ::getopt(argc, argv, "n:o:h")) != -1) {
		if (opt == 'n') {
			ngenerators = std::atoi(::optarg);
		} else if (opt == 'o') {
			filename = ::optarg;
		} else if (opt == 'h') {
			help_requested = true;
		}
	}
	if (help_requested) {
		usage(argv[0]);
		return 0;
	}
	if (optind != argc) {
		std::cerr << "No file argument is allowed." << std::endl;
		return 1;
	}
	if (ngenerators <= 0) {
		std::cerr << "Bad no. of generators: " << ngenerators << std::endl;
		return 1;
	}
	try {
		std::ofstream out(filename);
		out.exceptions(std::ios::failbit | std::ios::badbit);
		generate_mersenne_twisters(out, ngenerators);
	} catch (...) {
		std::cerr << "Bad output file: " << filename << std::endl;
		return 1;
	}
	std::clog
		<< "Saved "
		<< ngenerators
		<< " MT configuration(s) in \""
		<< filename << "\""
		<< std::endl;
	return 0;
}
