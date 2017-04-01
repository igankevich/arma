#include <unistd.h>       // for getopt, optarg, optind
#include <algorithm>      // for generate_n
#include <chrono>         // for duration, system_clock, system_clock::time...
#include <cstdlib>        // for atoi, size_t
#include <functional>     // for reference_wrapper, ref
#include <iostream>       // for operator<<, basic_ostream, clog, endl, ofs...
#include <fstream>        // for ofstream, ifstream
#include <iterator>       // for ostream_iterator
#include <string>         // for string, operator<<

#include "config.hh"
#include "parallel_mt.hh" // for parallel_mt_seq, mt_config (ptr only), ope...

void
generate_mersenne_twisters(std::string filename, size_t num_generators) {
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	arma::parallel_mt_seq<> seq(seed);
	std::ofstream out(filename);
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
		<< "USAGE: "
		<< (argv0 == nullptr ? "arma-dcmt" : argv0)
		<< " -n NUM_GENERATORS -o OUTFILE -h\n";
}

int
main(int argc, char* argv[]) {
	size_t ngenerators = 128;
	std::string filename = arma::config::mt_config_file;
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
		std::clog << "Bad file argument." << std::endl;
		return 1;
	}
	generate_mersenne_twisters(filename, ngenerators);
	std::clog
		<< "Saved "
		<< ngenerators
		<< " MT configs in \""
		<< filename << "\""
		<< std::endl;
	return 0;
}
