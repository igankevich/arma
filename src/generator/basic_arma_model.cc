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
		{"partition", sys::make_param(this->_partition, validate_shape<int,3>)},
		{"transform", sys::wrap_param(nit_wrapper(
			this->_nittransform,
			this->_linear
		))},
		{"acf", sys::wrap_param(acf_wrapper(this->_acf))},
		{"verification", sys::make_param(this->_vscheme)},
		{"order", sys::make_param(this->_order, validate_shape<int,3>)},
	};
}

template <class T>
arma::Shape3D
arma::generator::Basic_ARMA_model<T>::get_partition_shape(
	Shape3D order,
	int nprngs
) {
	Shape3D ret;
	if (blitz::product(this->_partition) > 0) {
		ret = this->_partition;
	} else {
		const Shape3D shape = this->_outgrid.size();
		const Shape3D guess1 = blitz::max(
			order * 2,
			Shape3D(10, 10, 10)
		);
		#if ARMA_OPENMP
		const int parallelism = std::min(omp_get_max_threads(), nprngs);
		#else
		const int parallelism = nprngs;
		#endif
		const int npar = std::max(1, 7*int(std::cbrt(parallelism)));
		const Shape3D guess2 = blitz::div_ceil(
			shape,
			Shape3D(npar, npar, npar)
		);
		ret = blitz::min(guess1, guess2) + blitz::abs(guess1 - guess2) / 2;
	}
	return ret;
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
	if (this->_vscheme == Verification_scheme::Manual) {
		bits::write_csv("acf.csv", this->_acf, this->_acf.grid());
	}
	{
		std::ofstream out("acf");
		out << this->_acf;
	}
	this->determine_coefficients();
	this->validate();
	Array3D<T> zeta = this->do_generate();
	if (!this->_linear) {
		this->_nittransform.transform_realisation(this->_acf, zeta);
	}
	return zeta;
}

#if ARMA_NONE || ARMA_OPENCL
#include "basic_arma_model_sequential.cc"
#elif ARMA_OPENMP
#include "basic_arma_model_parallel.cc"
#endif

template class arma::generator::Basic_ARMA_model<ARMA_REAL_TYPE>;
