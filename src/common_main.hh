#ifndef COMMON_MAIN_HH
#define COMMON_MAIN_HH

#include <exception>
#include <iostream>
#include <string>

#include <gsl/gsl_errno.h>

#if defined(ARMA_CLFFT)
#include <clFFT.h>
#endif

#include "config.hh"

#include "generator/ar_model.hh"
#include "generator/ma_model.hh"
#include "generator/arma_model.hh"
#include "generator/plain_wave_model.hh"
#include "generator/lh_model.hh"

#include "velocity/high_amplitude_solver.hh"
#include "velocity/linear_solver.hh"
#include "velocity/plain_wave_solver.hh"
#if defined(WITH_SMALL_AMPLITUDE_SOLVER)
#include "velocity/small_amplitude_solver.hh"
#endif

#if ARMA_OPENCL
#include "opencl/opencl.hh"
#endif

#include "profile.hh"
#if ARMA_PROFILE
#include "profile_counters.hh"
#endif

void
register_all_counters() {
	#if ARMA_PROFILE
	arma::register_counter(CNT_HARTS_G1, "harts_g1");
	arma::register_counter(CNT_HARTS_G2, "harts_g2");
	arma::register_counter(CNT_HARTS_FFT, "harts_fft");
	arma::register_counter(CNT_HARTS_COPY_TO_HOST, "harts_copy_to_host");
	arma::register_counter(CNT_COPY_TO_HOST, "copy_to_host");
	arma::register_counter(CNT_WRITE_SURFACE, "write_surface");
	#endif
}

void
print_exception_and_terminate() {
	if (std::exception_ptr ptr = std::current_exception()) {
		try {
			std::rethrow_exception(ptr);
		#if ARMA_OPENCL
		} catch (cl::Error err) {
			std::cerr << err << std::endl;
			std::abort();
		#endif
		} catch (const std::exception& e) {
			std::cerr << "ERROR: " << e.what() << std::endl;
			std::abort();
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
	std::cerr << "GSL error reason: " << reason << '.' << std::endl;
}

template <class T>
void
register_all_solvers(arma::ARMA_driver<T>& drv) {
	using namespace ::arma::velocity;
	drv.template register_solver<Linear_solver<T>>("linear");
	drv.template register_solver<Plain_wave_solver<T>>("plain");
	drv.template register_solver<High_amplitude_solver<T>>("high_amplitude");
	#if defined(WITH_SMALL_AMPLITUDE_SOLVER)
	drv.template register_solver<Small_amplitude_solver<T>>("small_amplitude");
	#endif
}

template <class T>
void
register_all_models(arma::ARMA_driver<T>& drv) {
	using namespace ::arma::generator;
	drv.template register_model<AR_model<T>>("AR");
	drv.template register_model<MA_model<T>>("MA");
	drv.template register_model<ARMA_model<T>>("ARMA");
	drv.template register_model<Plain_wave_model<T>>("plain_wave");
	drv.template register_model<Longuet_Higgins_model<T>>("LH");
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
