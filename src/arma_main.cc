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

	explicit
	ARMA_driver_kernel(const std::string& filename):
	_filename(filename) {
		register_all_models<T>(*this);
		register_all_solvers<T>(*this);
		this->open(this->_filename);
	}

	void
	act() override {
		this->echo_parameters();
		bsc::upstream(this, this->_model);
	}

	void
	react(bsc::kernel* child) {
		this->_zeta.reference(this->_model->zeta());
		#if ARMA_OPENCL
		this->_zeta.copy_to_host_if_exists();
		#endif
		this->_model->verify(this->_zeta);
		this->compute_velocity_potentials();
		this->write_all();
		bsc::commit(this);
	}

};
#endif

template <class T>
void
run_arma(const std::string& input_filename) {
	using namespace arma;
	#if ARMA_BSCHEDULER
	bsc::factory_guard g;
	if (bsc::this_application::is_master()) {
		bsc::send(new ARMA_driver_kernel<T>(input_filename));
	}
	bsc::wait_and_return();
	#else
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
	#endif
}
