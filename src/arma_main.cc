#include <gsl/gsl_errno.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <unistd.h>

#include "arma_driver.hh"
#include "errors.hh"
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
		<< "USAGE: "
		<< (argv0 == nullptr ? "arma" : argv0)
		<< " -c CONFIGFILE\n";
}

template<class Solver, class Driver>
void
register_vpsolver(Driver& drv, std::string key) {
	drv.template register_velocity_potential_solver<Solver>(key);
}

#if ARMA_OPENCL
void
init_opencl() {
	::arma::opencl::init();
}
#endif

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

	using namespace arma;

	/// floating point type (float, double, long double or multiprecision number
	/// C++ class)
	typedef ARMA_REAL_TYPE Real;

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
		/// input file with various driver parameters
		ARMA_driver<Real> driver;
		using namespace velocity;
		register_vpsolver<Linear_solver<Real>>(driver, "linear");
		register_vpsolver<Plain_wave_solver<Real>>(driver, "plain");
		register_vpsolver<High_amplitude_solver<Real>>(driver, "high_amplitude");
		#if defined(WITH_SMALL_AMPLITUDE_SOLVER)
		register_vpsolver<Small_amplitude_solver<Real>>(driver, "small_amplitude");
		#endif
		std::ifstream cfg(input_filename);
		if (!cfg.is_open()) {
			std::cerr << "Cannot open input file "
				"\"" << input_filename << "\"."
				<< std::endl;
			throw std::runtime_error("bad input file");
		}
		write_key_value(std::clog, "Input file", input_filename);
		cfg >> driver;
		try {
			driver.generate_wavy_surface();
			driver.compute_velocity_potentials();
			if (driver.vscheme() != Verification_scheme::No_verification) {
				driver.write_wavy_surface("zeta", Output_format::Blitz);
				driver.write_velocity_potentials("phi", Output_format::Blitz);
			}
			if (driver.vscheme() == Verification_scheme::Manual) {
				driver.write_wavy_surface("zeta.csv", Output_format::CSV);
				driver.write_velocity_potentials("phi.csv", Output_format::CSV);
			}
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
	#if ARMA_PROFILE
	arma::print_counters(std::clog);
	#endif
	return 0;
}
