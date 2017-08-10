#include "arma_driver.hh"
#include "bits/object_wrapper.hh"
#include "bits/write_csv.hh"
#include "params.hh"
#include "profile.hh"
#include "io/binary_stream.hh"
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <iostream>

template <class T>
arma::Grid<T,3>
arma::ARMA_driver<T>::velocity_potential_grid() const {
	const Grid<T,3> outgrid = this->_model->grid();
	const int nz = this->_solver->domain().num_points(1);
	const int nx = outgrid.num_points(1);
	const int ny = outgrid.num_points(2);
	return Grid<T,3>(
		{nz, nx, ny},
		{
			this->_solver->domain().length(1),
			outgrid.length(1),
			outgrid.length(2)
		}
	);
}

template <class T>
void
arma::ARMA_driver<T>::write_wavy_surface() {
	if (this->oflags().isset(Output_flags::Blitz)) {
		std::string filename = get_surface_filename(Output_flags::Blitz);
		std::ofstream(filename) << _zeta;
	} else if (this->oflags().isset(Output_flags::CSV)) {
		std::string filename = get_surface_filename(Output_flags::CSV);
		bits::write_csv(filename, _zeta, _zeta.grid());
	} else if (this->oflags().isset(Output_flags::Binary)) {
		std::string filename = get_surface_filename(Output_flags::Binary);
		io::Binary_stream out(filename);
		out.write(this->_zeta);
	}
}

template <class T>
void
arma::ARMA_driver<T>::write_velocity_potentials() {
	if (this->oflags().isset(Output_flags::Blitz)) {
		std::string filename = get_velocity_filename(Output_flags::Blitz);
		std::ofstream(filename) << this->_vpotentials;
	} else if (this->oflags().isset(Output_flags::CSV)) {
		std::string filename = get_velocity_filename(Output_flags::CSV);
		bits::write_4d_csv(
			filename,
			this->_vpotentials,
			this->_solver->domain(),
			this->_model->grid()
		);
	} else if (this->oflags().isset(Output_flags::Binary)) {
		// TODO
	}
}

template <class T>
void
arma::ARMA_driver<T>::generate_wavy_surface() {
	echo_parameters();
	this->_zeta.reference(this->_model->generate());
	#if ARMA_OPENCL
	this->_zeta.copy_to_host_if_exists();
	#endif
	this->_model->verify(this->_zeta);
}

template <class T>
void
arma::ARMA_driver<T>::compute_velocity_potentials() {
	_zeta.setgrid(this->wavy_surface_grid());
	this->_vpotentials.reference(_solver->operator()(_zeta));
}

template <class T>
std::istream&
arma::operator>>(std::istream& in, ARMA_driver<T>& rhs) {
	typedef typename ARMA_driver<T>::model_type model_type;
	typedef typename ARMA_driver<T>::vpsolver_type vpsolver_type;
	bits::Object_wrapper<model_type> model_wrapper(
		rhs._model,
		rhs._models
	);
	bits::Object_wrapper<vpsolver_type> vpsolver_wrapper(
		rhs._solver,
		rhs._solvers
	);
	sys::parameter_map params({
		{"model", sys::make_param(model_wrapper)},
		{"velocity_potential_solver", sys::make_param(vpsolver_wrapper)},
	});
	in >> params;
	if (!rhs._solver) {
		std::cerr
			<< "Bad \"velocity_potential_solver\": null"
			<< std::endl;
		throw std::runtime_error("bad solver");
	}
	if (!rhs._model) {
		std::cerr
			<< "Bad \"generator\": null"
			<< std::endl;
		throw std::runtime_error("bad generator");
	}
	return in;
}

template <class T>
void
arma::ARMA_driver<T>::echo_parameters() {
	if (this->_model) {
		write_key_value(
			std::clog,
			"Output grid size",
			this->_model->grid().size()
		);
		write_key_value(
			std::clog,
			"Output grid patch size",
			this->_model->grid().patch_size()
		);
		write_key_value(std::clog, "Model", *this->_model);
	}
	if (this->_solver) {
		write_key_value(
			std::clog,
			"Velocity potential solver name",
			typeid(*this->_solver).name()
		);
		write_key_value(
			std::clog,
			"Velocity potential solver",
			*_solver
		);
	}
}

template <class T>
void
arma::ARMA_driver<T>::write_all() {
	ARMA_PROFILE_START(write_all);
	if (this->oflags().isset(Output_flags::Surface)) {
		this->write_wavy_surface();
		this->write_velocity_potentials();
	}
	ARMA_PROFILE_END(write_all);
}


template <class T>
void
arma::ARMA_driver<T>::open(const std::string& filename) {
	std::ifstream cfg(filename);
	if (!cfg.is_open()) {
		std::cerr << "Cannot open input file "
			"\"" << filename << "\"."
			<< std::endl;
		throw std::runtime_error("bad input file");
	}
	write_key_value(std::clog, "Input file", filename);
	cfg >> *this;
}

template class arma::ARMA_driver<ARMA_REAL_TYPE>;
template std::istream&
arma::operator>>(std::istream& in, ARMA_driver<ARMA_REAL_TYPE>& m);
