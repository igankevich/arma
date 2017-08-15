#include <cstdlib>
#include <unistd.h>

#include "arma_driver.hh"
#include "errors.hh"
#include "common_main.hh"

template <class T>
void
run_arma(const std::string& input_filename) {
	using namespace arma;
	/// input file with various driver parameters
	ARMA_driver<T> driver;
	register_all_models<T>(driver);
	register_all_solvers<T>(driver);
	try {
		driver.open(input_filename);
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
