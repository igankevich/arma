#include <gsl/gsl_errno.h>   // for gsl_set_error_handler
#include <cstdlib>           // for exit
#include <exception>         // for exception, exception_ptr, current_ex...
#include <iostream>          // for operator<<, basic_ostream, cerr, endl
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
main() {

	/// Print GSL errors and proceed execution.
	/// Throw domain-specific exception later.
	gsl_set_error_handler(print_error_and_continue);
	std::set_terminate(print_exception_and_terminate);

	using namespace autoreg;

	/// floating point type (float, double, long double or multiprecision number
	/// C++ class)
	typedef float Real;

	/// input file with various model parameters
	const char* input_filename = "autoreg.model";
	Autoreg_model<Real> model;
	std::ifstream cfg(input_filename);
	cfg >> model;
	model.act();
	return 0;
}
