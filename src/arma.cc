#include <gsl/gsl_errno.h>   // for gsl_set_error_handler
#include <cstdlib>           // for exit
#include <exception>         // for exception, exception_ptr, current_ex...
#include <iostream>          // for operator<<, basic_ostream, cerr, endl
#include <unistd.h>          // for getopt
#include "arma_driver.hh"    // for Autoreg_model, operator>>

void
print_exception_and_terminate() {
	if (std::exception_ptr ptr = std::current_exception()) {
		try {
			std::rethrow_exception(ptr);
		} catch (const std::exception& e) {
			std::cerr << "ERROR: " << e.what() << std::endl;
		} catch (...) { std::cerr << "UNKNOWN ERROR. Aborting." << std::endl; }
	}
	std::exit(1);
}

void
print_error_and_continue(const char* reason, const char* file, int line,
                         int gsl_errno) {
	std::cerr << "GSL error reason: " << reason << '.' << std::endl;
}

void
usage(char* argv0) {
	std::cout
		<< "USAGE: "
		<< (argv0 == nullptr ? "arma" : argv0)
		<< " -c CONFIGFILE\n";
}

int
main(int argc, char* argv[]) {

	/// Print GSL errors and proceed execution.
	/// Throw domain-specific exception later.
	gsl_set_error_handler(print_error_and_continue);
	std::set_terminate(print_exception_and_terminate);

	using namespace arma;

	/// floating point type (float, double, long double or multiprecision number
	/// C++ class)
	typedef float Real;

	std::string input_filename;
	bool help_requested = false;
	int opt = 0;
	while ((opt = ::getopt(argc, argv, "c:h")) != -1) {
		switch (opt) {
			case 'c':
				input_filename = ::optarg;
				break;
			case 'h':
				help_requested = true;
				break;
		}
	}

	if (help_requested || input_filename.empty()) {
		usage(argv[0]);
	} else {
		/// input file with various model parameters
		Autoreg_model<Real> model;
		std::ifstream cfg(input_filename);
		if (!cfg.is_open()) {
			std::clog << "Cannot open input file \"" << input_filename << "\"." << std::endl;
			throw std::runtime_error("bad input file");
		}
		write_key_value(std::clog, "Input file", input_filename);
		cfg >> model;
		try {
			model.act();
		} catch (const prng_error& err) {
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
	return 0;
}
