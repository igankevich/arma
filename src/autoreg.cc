#include <gsl/gsl_errno.h>   // for gsl_set_error_handler
#include <cstdlib>           // for exit
#include <exception>         // for exception, exception_ptr, current_ex...
#include <iostream>          // for operator<<, basic_ostream, cerr, endl
#include <unistd.h>          // for getopt
#include "autoreg_driver.hh" // for Autoreg_model, operator>>

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

int
main(int argc, char* argv[]) {

	/// Print GSL errors and proceed execution.
	/// Throw domain-specific exception later.
	gsl_set_error_handler(print_error_and_continue);
	std::set_terminate(print_exception_and_terminate);

	using namespace autoreg;

	/// floating point type (float, double, long double or multiprecision number
	/// C++ class)
	typedef float Real;

	std::string input_filename = "input.autoreg";
	int opt = 0;
	while ((opt = ::getopt(argc, argv, "c:")) != -1) {
		if (opt == 'c') {
			input_filename = ::optarg;
		}
	}
	write_key_value(std::clog, "Input file", input_filename);

	/// input file with various model parameters
	Autoreg_model<Real> model;
	std::ifstream cfg(input_filename);
	if (!cfg.is_open()) {
		std::clog << "Cannot open input file \"" << input_filename << "\"." << std::endl;
		throw std::runtime_error("bad input file");
	}
	cfg >> model;
	model.act();
	return 0;
}
