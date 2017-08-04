#include "arma.hh"
#include "basic_arma_model.hh"
#include "bits/acf_wrapper.hh"
#include "bits/transform_wrapper.hh"
#include "bits/write_csv.hh"
#include "nonlinear/nit_transform.hh"
#include "validators.hh"
#include "util.hh"
#include "profile_counters.hh"
#include "white_noise.hh"

#include <random>

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
arma::generator::Basic_ARMA_model<T>::generate_white_noise() {
	const T var_wn = this->white_noise_variance();
	write_key_value(std::clog, "White noise variance", var_wn);
	if (var_wn < T(0)) {
		throw std::invalid_argument("variance is less than zero");
	}
	return prng::generate_white_noise<T,3>(
		this->grid().num_points(),
		this->_noseed,
		std::normal_distribution<T>(T(0), std::sqrt(var_wn))
	);
}

template <class T>
arma::Array3D<T>
arma::generator::Basic_ARMA_model<T>::generate() {
	ARMA_PROFILE_CNT(CNT_NIT,
		if (!this->_linear) {
			auto copy = this->_acf.copy();
			this->_nittransform.transform_ACF(copy);
		}
	);
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
	ARMA_PROFILE_BLOCK("determine_coefficients",
		this->determine_coefficients();
	);
	ARMA_PROFILE_BLOCK("validate",
		this->validate();
	);
	Array3D<T> zeta;
	ARMA_PROFILE_BLOCK("generate_surface",
		zeta.reference(this->do_generate());
	);
	ARMA_PROFILE_CNT(CNT_NIT,
		if (!this->_linear) {
			this->_nittransform.transform_realisation(this->_acf, zeta);
		}
	);
	return zeta;
}

template class arma::generator::Basic_ARMA_model<ARMA_REAL_TYPE>;
