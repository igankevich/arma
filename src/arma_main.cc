#include <gsl/gsl_errno.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <unistd.h>

#include "arma_driver.hh"
#include "errors.hh"
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

void
usage(char* argv0) {
	std::cout
		<< "usage: "
		<< (argv0 == nullptr ? "arma" : argv0)
		<< " [-h] configfile\n";
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

#if ARMA_OPENCL
void
init_opencl() {
	::arma::opencl::init();
}
#endif

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

	#if ARMA_PROFILE
	arma::register_all_counters();
	#endif

	/// Print GSL errors and proceed execution.
	/// Throw domain-specific exception later.
	gsl_set_error_handler(print_error_and_continue);
	std::set_terminate(print_exception_and_terminate);

	#if ARMA_OPENCL
	init_opencl();
	#endif

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
