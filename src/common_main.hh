#ifndef COMMON_MAIN_HH
#define COMMON_MAIN_HH

#include <exception>
#include <iostream>

#include <gsl/gsl_errno.h>

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

#if ARMA_PROFILE
#include "profile_counters.hh"
#endif

void
register_all_counters() {
	#if ARMA_PROFILE
	arma::register_counter(CNT_WINDOWFUNC, "window_function");
	arma::register_counter(CNT_SECONDFUNC, "second_function");
	arma::register_counter(CNT_FFT, "fft");
	arma::register_counter(CNT_DEVTOHOST_COPY, "dev_to_host_copy");
	arma::register_counter(CNT_COPY_TO_HOST, "copy_to_host");
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

void
arma_init() {
	#if ARMA_PROFILE
	register_all_counters();
	#endif

	/// Print GSL errors and proceed execution.
	/// Throw domain-specific exception later.
	gsl_set_error_handler(print_error_and_continue);
	std::set_terminate(print_exception_and_terminate);

	#if ARMA_OPENCL
	::arma::opencl::init();
	#endif
}

#endif // COMMON_MAIN_HH
