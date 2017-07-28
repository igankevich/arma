#include "arma.hh"
#include "basic_arma_model.hh"
#include "bits/acf_wrapper.hh"
#include "bits/transform_wrapper.hh"
#include "bits/write_csv.hh"
#include "nonlinear/nit_transform.hh"
#include "validators.hh"
#include "util.hh"

#include <random>
#if ARMA_OPENMP
#include <omp.h>
#endif

template <class T>
sys::parameter_map::map_type
arma::generator::Basic_ARMA_model<T>::parameters() {
	typedef bits::Transform_wrapper<transform_type> nit_wrapper;
	typedef bits::ACF_wrapper<T> acf_wrapper;
	return {
		{"out_grid", sys::make_param(this->_outgrid, validate_grid<T,3>)},
		{"no_seed", sys::make_param(this->_noseed)},
		{"transform", sys::wrap_param(nit_wrapper(
			this->_nittransform,
			this->_linear
		))},
		{"acf", sys::wrap_param(acf_wrapper(this->_acf))},
		{"output", sys::make_param(this->_oflags)},
		{"order", sys::make_param(this->_order, validate_shape<int,3>)},
	};
}

#include "basic_arma_model_verify.cc"

template <class T>
arma::Array3D<T>
arma::generator::Basic_ARMA_model<T>::generate() {
	if (!this->_linear) {
		auto copy = this->_acf.copy();
		this->_nittransform.transform_ACF(copy);
	}
	arma::write_key_value(std::clog, "ACF variance", ACF_variance(this->_acf));
	if (this->_oflags.isset(Output_flags::ACF)) {
		if (this->_oflags.isset(Output_flags::CSV)) {
			bits::write_csv("acf.csv", this->_acf, this->_acf.grid());
		}
		if (this->_oflags.isset(Output_flags::Blitz)) {
			std::ofstream out("acf");
			out << this->_acf;
		}
	}
	this->determine_coefficients();
	this->validate();
	Array3D<T> zeta = this->do_generate();
	if (!this->_linear) {
		this->_nittransform.transform_realisation(this->_acf, zeta);
	}
	return zeta;
}

template class arma::generator::Basic_ARMA_model<ARMA_REAL_TYPE>;
