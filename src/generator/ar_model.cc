#include "ar_model.hh"

#include "linalg.hh"
#include "params.hh"
#include "util.hh"
#include "util.hh"
#include "voodoo.hh"
#include "yule_walker.hh"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>

#include "ar_model_surf.cc"

#if ARMA_OPENMP || ARMA_OPENCL || ARMA_BSCHEDULER
namespace {

	arma::Shape3D
	get_partition_shape(
		arma::Shape3D partition_shape,
		arma::Shape3D grid_shape,
		arma::Shape3D order,
		int parallelism
	) {
		using blitz::product;
		using std::min;
		using std::cbrt;
		using arma::Shape3D;
		Shape3D ret;
		if (product(partition_shape) > 0) {
			ret = partition_shape;
		} else {
			const Shape3D guess1 = blitz::max(
				order * 2,
				Shape3D(10, 10, 10)
			                       );
			const int npar = std::max(1, 7*int(cbrt(parallelism)));
			const Shape3D guess2 = blitz::div_ceil(
				grid_shape,
				Shape3D(npar, npar, npar)
			                       );
			ret = blitz::min(guess1, guess2) + blitz::abs(guess1 - guess2) / 2;
		}
		return ret;
	}

}
#endif

template <class T>
T
arma::generator::AR_model<T>
::white_noise_variance(Array3D<T> phi) const {
	blitz::RectDomain<3> subdomain(Shape3D(0, 0, 0), phi.shape() - 1);
	return this->_acf(0,0,0) - blitz::sum(phi * this->_acf(subdomain));
}

template <class T>
void
arma::generator::AR_model<T>
::validate() const {
	validate_process(this->_phi);
}

template <class T>
void
arma::generator::AR_model<T>
::generate_surface(
	Array3D<T>& zeta,
	const Domain3D& subdomain
) {
	ar_generate_surface(zeta, this->_phi, subdomain);
}

template <class T>
void
arma::generator::AR_model<T>
::determine_coefficients() {
	switch (this->_algorithm) {
	case AR_algorithm::Gauss_elimination:
		this->determine_coefficients_gauss();
		break;
	case AR_algorithm::Choi:
		this->determine_coefficients_choi();
		break;
	default:
		throw std::runtime_error("bad AR algorithm");
	}
}

template <class T>
void
arma::generator::AR_model<T>
::determine_coefficients_gauss() {
	using blitz::all;
	if (!all(this->order() <= this->_acf.shape())) {
		throw std::runtime_error("bad AR model order");
	}
	using blitz::Range;
	using blitz::toEnd;
	AC_matrix_generator<T> generator(this->_acf, this->order());
	Array2D<T> acm = generator();
	const int m = acm.rows() - 1;
	/**
	   eliminate the first equation and move the first column of the
	   remaining matrix to the right-hand side of the system
	 */
	Array1D<T> rhs(m);
	rhs = acm(Range(1, toEnd), 0);

	// lhs is the autocovariance matrix without first
	// column and row
	Array2D<T> lhs(blitz::shape(m, m));
	lhs = acm(Range(1, toEnd), Range(1, toEnd));
	acm.free();

	assert(lhs.extent(0) == m);
	assert(lhs.extent(1) == m);
	assert(rhs.extent(0) == m);
	assert(linalg::is_symmetric(lhs));
	assert(linalg::is_positive_definite(lhs));
	linalg::cholesky(lhs, rhs);
	assert(this->_phi.numElements() == rhs.numElements() + 1);
	if (this->_phi.numElements() > 1) {
		this->_phi(0,0,0) = 0;
	}
	std::copy_n(rhs.data(), rhs.numElements(), this->_phi.data() + 1);
	this->_varwn = T(2)*this->white_noise_variance(this->_phi);
	{ std::ofstream("gauss") << this->_phi; }
}

template <class T>
void
arma::generator::AR_model<T>
::determine_coefficients_choi() {
	using blitz::all;
	using blitz::max;
	Yule_walker_solver<T> solver(this->_acf);
	solver.var_epsilon(T(1e-6));
	if (all(this->_order) > 0) {
		solver.max_order(max(this->_order));
	}
	this->_phi.reference(solver.solve());
	this->_order = this->_phi.shape();
//	this->_varwn = solver.white_noise_variance();
	this->_varwn = this->white_noise_variance(this->_phi);
	write_key_value(std::clog, "New AR model order", this->_order);
	{ std::ofstream("choi") << this->_phi; }
}

template <class T>
void
arma::generator::AR_model<T>
::read(std::istream& in) {
	typedef typename Basic_model<T>::grid_type grid_type;
	sys::parameter_map params {
		{
			{"algorithm", sys::make_param(this->_algorithm)},
			{"partition", sys::make_param(
				 this->_partition,
				 validate_shape<int, 3>
			 )},
		},
		true
	};
	params.insert(this->parameters());
	in >> params;
	// resize output grid to match ACF delta size
	this->_outgrid =
		grid_type(
			this->_outgrid.num_points(),
			this->_acf.grid().delta() * this->_outgrid.num_patches() * T(1.0)
		);
	this->_phi.resize(this->order());
}

template <class T>
void
arma::generator::AR_model<T>
::write(std::ostream& out) const {
	out << "grid=" << this->grid()
	    << ",order=" << this->order()
	    << ",output=" << this->_oflags
	    << ",acf.shape=" << this->_acf.shape()
	    << ",transform=" << this->_nittransform
	    << ",noseed=" << this->_noseed;
}

#if ARMA_NONE
#include "ar_model_sequential.cc"
#elif ARMA_OPENMP
#include "ar_model_parallel.cc"
#elif ARMA_OPENCL
#include "ar_model_opencl.cc"
#elif ARMA_BSCHEDULER
#include "ar_model_bscheduler.cc"
#endif

template class arma::generator::AR_model<ARMA_REAL_TYPE>;
