#include <gsl/gsl_errno.h>   // for gsl_set_error_handler
#include <cstdlib>           // for exit
#include <exception>         // for exception, exception_ptr, current_ex...
#include <iostream>          // for operator<<, basic_ostream, cerr, endl
#include <unistd.h>          // for getopt
#include "arma_driver.hh"    // for ARMA_driver, operator>>
#include "velocity/linear_velocity_potential_field.hh"
#include "velocity/plain_wave_velocity_field.hh"
#include "velocity/high_amplitude_velocity_potential_field.hh"

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

int
main(int argc, char* argv[]) {

	/// Print GSL errors and proceed execution.
	/// Throw domain-specific exception later.
	gsl_set_error_handler(print_error_and_continue);
	std::set_terminate(print_exception_and_terminate);

	using namespace arma;

	/// floating point type (float, double, long double or multiprecision number
	/// C++ class)
	typedef double Real;

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
		typedef Linear_velocity_potential_field<Real> linear_solver;
		typedef Plain_wave_velocity_field<Real> plain_wave_solver;
		typedef High_amplitude_velocity_potential_field<Real> highamp_solver;
		/// input file with various driver parameters
		ARMA_driver<Real> driver;
		register_vpsolver<linear_solver>(driver, "linear");
		register_vpsolver<plain_wave_solver>(driver, "plain");
		register_vpsolver<highamp_solver>(driver, "high_amplitude");
		std::ifstream cfg(input_filename);
		if (!cfg.is_open()) {
			std::clog << "Cannot open input file "
				"\"" << input_filename << "\"."
				<< std::endl;
			throw std::runtime_error("bad input file");
		}
		write_key_value(std::clog, "Input file", input_filename);
		cfg >> driver;
		try {
			driver.generate_wavy_surface();
			driver.compute_velocity_potentials();
			driver.write_wavy_surface("zeta", Output_format::Blitz);
			driver.write_velocity_potentials("phi", Output_format::Blitz);
			if (driver.vscheme() == Verification_scheme::Manual) {
				driver.write_wavy_surface("zeta.csv", Output_format::CSV);
				driver.write_velocity_potentials("phi.csv", Output_format::CSV);
			}
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
