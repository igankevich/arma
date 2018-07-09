#ifndef COMMON_MAIN_HH
#define COMMON_MAIN_HH

#include <exception>
#include <iostream>
#include <string>

#include <gsl/gsl_errno.h>

#include "register_all.hh"

void
print_exception_and_terminate() {
	if (std::exception_ptr ptr = std::current_exception()) {
		try {
			std::rethrow_exception(ptr);
		#if ARMA_OPENCL
		} catch (const cl::Error& err) {
			std::cerr << err << std::endl;
			std::abort();
		#endif
		} catch (const std::exception& e) {
			std::cerr << "ERROR: " << e.what() << std::endl;
			std::exit(1);
		} catch (...) {
			std::cerr << "UNKNOWN ERROR. Aborting." << std::endl;
		}
	}
	std::abort();
}

void
print_error_and_continue(
	const char* reason,
	const char* file,
	int line,
	int gsl_errno
) {
	std::cerr << "GSL error: " << file << ':' << reason << '.' << std::endl;
	throw std::runtime_error(reason);
}

#if ARMA_OPENGL
void
init_opengl(int argc, char* argv[]);
#endif

void
arma_init(int argc, char* argv[]) {
	#if ARMA_PROFILE
	register_all_counters();
	#endif
	/// Print GSL errors and proceed execution.
	/// Throw domain-specific exception later.
	gsl_set_error_handler(print_error_and_continue);
	std::set_terminate(print_exception_and_terminate);
	#if ARMA_OPENGL
	init_opengl(argc, argv);
	#endif
	#if ARMA_OPENCL
	::arma::opencl::init();
	#endif
}

void
arma_finalise() {
	#if ARMA_PROFILE
	arma::print_counters(std::clog);
	#endif
	#if defined(ARMA_CLFFT)
	#define CHECK(x) ::cl::detail::errHandler((x), #x);
	CHECK(clfftTeardown());
	#undef CHECK
	#endif
}

void
usage(char* argv0) {
	std::cout
		<< "usage: "
		<< (argv0 == nullptr ? "arma" : argv0)
		<< " [-h] INPUTFILE\n";
}

template <class T>
void
run_arma(const std::string& input_filename);

int
main(int argc, char* argv[]) {
	ARMA_EVENT_START("programme", "main", 0);
	arma_init(argc, argv);
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
	arma_finalise();
	ARMA_EVENT_END("programme", "main", 0);
	return 0;
}

#endif // COMMON_MAIN_HH
