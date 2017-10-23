#include <cstdlib>
#include <unistd.h>

#if ARMA_BSCHEDULER
#include <bscheduler/api.hh>
#include <bscheduler/base/error_handler.hh>
#endif

#include "arma_driver.hh"
#include "errors.hh"
#include "common_main.hh"

#if ARMA_BSCHEDULER
template <class T>
class ARMA_driver_kernel: public bsc::kernel, public arma::ARMA_driver<T> {

public:
	using typename arma::ARMA_driver<T>::model_type;

private:
	std::string _filename;

public:

	ARMA_driver_kernel() {
		register_all_models<T>(*this);
		register_all_solvers<T>(*this);
	}

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
		this->_model->setf(bsc::kernel_flag::carries_parent);
		bsc::upstream<bsc::Remote>(this, this->_model);
	}

	void
	react(bsc::kernel* child) override {
		if (child != this->_model) {
			this->_model = dynamic_cast<model_type*>(child);
		}
		this->_zeta.reference(this->_model->zeta());
		#if ARMA_OPENCL
		this->_zeta.copy_to_host_if_exists();
		#endif
		this->_model->verify(this->_zeta);
		this->compute_velocity_potentials();
		this->write_all();
		bsc::commit(this);
	}

	void
	write(sys::pstream& out) const override {
		bsc::kernel::write(out);
		out << this->_zeta;
		out << this->_vpotentials;
		out << this->_filename;
		out << this->_solvername;
		out << *this->_solver;
	}

	void
	read(sys::pstream& in) override {
		bsc::kernel::read(in);
		in >> this->_zeta;
		in >> this->_vpotentials;
		in >> this->_filename;
		in >> this->_solvername;
		if (!this->_solvername.empty()) {
			auto result = this->_solvers.find(this->_solvername);
			if (result == this->_solvers.end()) {
				throw std::invalid_argument("bad solver name");
			}
			this->_solver = result->second();
			in >> *this->_solver;
		}
	}

};
#endif

template <class T>
void
run_arma(const std::string& input_filename) {
	using namespace arma;
	#if ARMA_BSCHEDULER
	bsc::install_error_handler();
	bsc::register_type<generator::AR_model<ARMA_REAL_TYPE> >();
	bsc::register_type<ARMA_driver_kernel<ARMA_REAL_TYPE> >();
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
