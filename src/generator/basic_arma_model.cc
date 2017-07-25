#include "basic_arma_model.hh"
#include "arma.hh"
#include <random>
#if ARMA_OPENMP
#include <omp.h>
#endif

template <class T>
arma::Shape3D
arma::generator::Basic_ARMA_model<T>::get_partition_shape(
	Shape3D order,
	int nprngs
) {
	Shape3D ret;
	if (blitz::product(this->_partition) > 0) {
		ret = _partition;
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

template <class T>
arma::Array3D<T>
arma::generator::Basic_ARMA_model<T>::generate() {
	Discrete_function<T,3> acf = this->acf();
	if (!this->_linear) {
		auto copy = acf.copy();
		this->_nittransform.transform_ACF(copy);
	}
	this->determine_coefficients();
	this->validate();
	Array3D<T> zeta = this->do_generate();
	if (!this->_linear) {
		this->_nittransform.transform_realisation(acf, zeta);
	}
	return zeta;
}

#if ARMA_NONE || ARMA_OPENCL
#include "basic_arma_model_sequential.cc"
#elif ARMA_OPENMP
#include "basic_arma_model_parallel.cc"
#endif

template class arma::generator::Basic_ARMA_model<ARMA_REAL_TYPE>;
