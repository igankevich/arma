#include <cstdlib>
#include <unistd.h>

#if ARMA_BSCHEDULER
#include <bscheduler/api.hh>
#endif

#include "arma_driver.hh"
#include "errors.hh"
#include "common_main.hh"

#if ARMA_BSCHEDULER
template <class T>
class ARMA_driver_kernel: public bsc::kernel, public arma::ARMA_driver<T> {

private:
	std::string _filename;

public:

	ARMA_driver_kernel(const std::string& filename):
	_filename(filename) {
		register_all_models<T>(*this);
		register_all_solvers<T>(*this);
		this->open(this->_filename);
	}

	void
	act() override {
		this->generate_wavy_surface();
		this->compute_velocity_potentials();
		this->write_all();
	}

};
#endif

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
