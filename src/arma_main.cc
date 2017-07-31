#include <cstdlib>
#include <unistd.h>

#include "arma_driver.hh"
#include "errors.hh"
#include "common_main.hh"

void
usage(char* argv0) {
	std::cout
		<< "usage: "
		<< (argv0 == nullptr ? "arma" : argv0)
		<< " [-h] configfile\n";
}

template <class T>
void
run_arma(const std::string& input_filename) {
	using namespace arma;
	/// input file with various driver parameters
	ARMA_driver<T> driver;
	register_all_models<T>(driver);
	register_all_solvers<T>(driver);
	driver.open(input_filename);
	try {
		driver.generate_wavy_surface();
		driver.compute_velocity_potentials();
		driver.write_all();
	} catch (const PRNG_error& err) {
		if (err.ngenerators() == 0) {
			std::cerr << "No parallel Mersenne Twisters configuration is found. "
				"Please, generate sufficient number of MTs with dcmt programme."
				<< std::endl;
		} else {
			std::cerr << "Insufficient number of parallel Mersenne Twisters found. "
				"Please, generate at least " << err.nparts() << " MTs for this run."
				<< std::endl;
		}
	}
}

int
main(int argc, char* argv[]) {
	arma_init();
	std::string input_filename;
	bool help_requested = false;
	int opt = 0;
	while ((opt = ::getopt(argc, argv, "h")) != -1) {
		switch (opt) {
			case 'h':
				help_requested = true;
				break;
		}
	}
	if (argc - ::optind > 1) {
		std::cerr << "Only one file argument is allowed." << std::endl;
		return 1;
	}

	if (input_filename.empty() && ::optind < argc) {
		input_filename = argv[::optind];
	}

	if (help_requested || input_filename.empty()) {
		usage(argv[0]);
	} else {
		/// floating point type (float, double, long double or multiprecision number
		/// C++ class)
		typedef ARMA_REAL_TYPE T;
		run_arma<T>(input_filename);
	}
	#if ARMA_PROFILE
	arma::print_counters(std::clog);
	#endif
	return 0;
}
