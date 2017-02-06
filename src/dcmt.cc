#include <unistd.h>       // for getopt, optarg, optind
#include <algorithm>      // for generate_n
#include <chrono>         // for duration, system_clock, system_clock::time...
#include <cstdlib>        // for atoi, size_t
#include <functional>     // for reference_wrapper, ref
#include <iostream>       // for operator<<, basic_ostream, clog, endl, ofs...
#include <fstream>        // for ofstream, ifstream
#include <iterator>       // for ostream_iterator
#include <string>         // for string, operator<<
#include "parallel_mt.hh" // for parallel_mt_seq, mt_config (ptr only), ope...

void
generate_mersenne_twisters(std::string filename, size_t num_generators) {
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	arma::parallel_mt_seq<> seq(seed);
	std::ofstream out(filename);
	std::ostream_iterator<arma::mt_config> out_it(out);
	std::generate_n(out_it, num_generators, std::ref(seq));
}

int
main(int argc, char* argv[]) {
	size_t ngenerators = 100;
	std::string filename = "MersenneTwister.dat";
	int opt = 0;
	while ((opt = ::getopt(argc, argv, "n:o:")) != -1) {
		if (opt == 'n') {
			ngenerators = std::atoi(::optarg);
		} else if (opt == 'o') {
			filename = ::optarg;
		}
	}
	if (optind != argc) {
		std::clog << "Bad file argument." << std::endl;
		return 1;
	}
	generate_mersenne_twisters(filename, ngenerators);
	std::clog << "Save " << ngenerators << " MT configs in " << filename
	          << std::endl;
	return 0;
}
