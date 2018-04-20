#include "basic_arma_model.hh"

#include "arma.hh"
#include "bits/acf_wrapper.hh"
#include "bits/transform_wrapper.hh"
#include "bits/write_csv.hh"
#include "nonlinear/nit_transform.hh"
#include "profile_counters.hh"
#include "util.hh"
#include "validators.hh"
#include "white_noise.hh"

#include <random>

template <class T>
sys::parameter_map::map_type
arma::generator::Basic_ARMA_model<T>::parameters() {
	typedef bits::Transform_wrapper<transform_type> nit_wrapper;
	return {
		{"out_grid", sys::make_param(this->_outgrid, validate_grid<T,3>)},
		{"no_seed", sys::make_param(this->_noseed)},
		{"transform", sys::wrap_param(nit_wrapper(
			this->_nittransform,
			this->_linear
		))},
		{"acf", sys::make_param(this->_acfgen)},
		{"output", sys::make_param(this->_oflags)},
		{"order", sys::make_param(this->_order, validate_shape<int,3>)},
		{"validate", sys::make_param(this->_validate)},
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
	using blitz::all;
	using constants::_2pi;
	typedef typename Basic_model<T>::grid_type grid_type;
	ARMA_PROFILE_BLOCK("generate_acf",
		this->_acf.reference(this->_acfgen.generate());
		if (all(this->_order) == 0) {
			this->_order = this->_acf.shape();
		}
		// resize output grid to match ACF delta size
		this->_outgrid =
			grid_type(
				this->_outgrid.num_points(),
				this->_acf.grid().delta() * this->_outgrid.num_patches() * _2pi<T>
			);
		write_key_value(std::clog, "Output grid", this->grid());
		write_key_value(
			std::clog,
			"Output grid patch size",
			this->grid().patch_size()
		);
	);
	ARMA_PROFILE_BLOCK("nit_acf",
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
		if (this->_validate) {
			this->validate();
		}
	);
	Array3D<T> zeta;
	ARMA_PROFILE_BLOCK("generate_surface",
		zeta.reference(this->do_generate());
	);
	// compensate for not using exponents in ACF
	using arma::stats::variance;
	using std::sqrt;
	using blitz::RectDomain;
	zeta *= sqrt(this->_acf(0,0,0) / variance(zeta(RectDomain<3>(zeta.shape()/2, zeta.shape()-1))));
	ARMA_PROFILE_BLOCK("nit_realisation",
		if (!this->_linear) {
			this->_nittransform.transform_realisation(this->_acf, zeta);
		}
	);
	return zeta;
}

#if ARMA_BSCHEDULER
template <class T>
void
arma::generator::Basic_ARMA_model<T>::act() {
	ARMA_PROFILE_BLOCK("nit_acf",
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
}

template <class T>
void
arma::generator::Basic_ARMA_model<T>::react(bsc::kernel*) {
	ARMA_PROFILE_BLOCK("nit_realisation",
		if (!this->_linear) {
			this->_nittransform.transform_realisation(this->_acf, this->_zeta);
		}
	);
}

template <class T>
arma::Array3D<T>
arma::generator::Basic_ARMA_model<T>::do_generate() {
	throw std::runtime_error("bad method");
}

template <class T>
void
arma::generator::Basic_ARMA_model<T>
::write(sys::pstream& out) const {
	Basic_model<T>::write(out);
	out << this->_acf;
	out << this->_order;
	out << this->_linear;
}

template <class T>
void
arma::generator::Basic_ARMA_model<T>
::read(sys::pstream& in) {
	Basic_model<T>::read(in);
	in >> this->_acf;
	in >> this->_order;
	in >> this->_linear;
}

#endif

template class arma::generator::Basic_ARMA_model<ARMA_REAL_TYPE>;
