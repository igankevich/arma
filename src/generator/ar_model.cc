#include "ar_model.hh"

#include "voodoo.hh"
#include "linalg.hh"
#include "params.hh"
#include "util.hh"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <random>

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
arma::generator::AR_model<T>::white_noise_variance(Array3D<T> phi) const {
	blitz::RectDomain<3> subdomain(Shape3D(0, 0, 0), phi.shape() - 1);
	return this->_acf(0,0,0) - blitz::sum(phi * this->_acf(subdomain));
}

template <class T>
void
arma::generator::AR_model<T>::validate() const {
	validate_process(this->_phi);
}

template <class T>
void
arma::generator::AR_model<T>::generate_surface(
	Array3D<T>& zeta,
	const Domain3D& subdomain
) {
	ar_generate_surface(zeta, this->_phi, subdomain);
}

template <class T>
void
arma::generator::AR_model<T>::determine_coefficients_old(bool do_least_squares) {
	std::clog << "id = " << this->id() << std::endl;
	using blitz::all;
	if (!all(this->order() <= this->_acf.shape())) {
		std::cerr << "AR model order is larger than ACF "
					 "size:\n\tAR model "
					 "order = "
				  << this->order() << "\n\tACF size = " << this->_acf.shape()
				  << std::endl;
		throw std::runtime_error("bad AR model order");
	}
	//_acf = _acf / _acf(0, 0, 0);
	using blitz::Range;
	using blitz::toEnd;
	// normalise Array3D to prevent big numbers when multiplying
	// matrices
	std::function<Array2D<T>()> generator;
	if (do_least_squares) {
		generator = AC_matrix_generator_LS<T>(this->_acf, this->order());
	} else {
		generator = AC_matrix_generator<T>(this->_acf, this->order());
	}
	Array2D<T> acm = generator();
	const int m = acm.rows() - 1;
	/**
	eliminate the first equation and move the first column of the
	remaining
	matrix to the right-hand side of the system
	*/
	Array1D<T> rhs(m);
	rhs = acm(Range(1, toEnd), 0);

	// lhs is the autocovariance matrix without first
	// column and row
	Array2D<T> lhs(blitz::shape(m, m));
	lhs = acm(Range(1, toEnd), Range(1, toEnd));

	assert(lhs.extent(0) == m);
	assert(lhs.extent(1) == m);
	assert(rhs.extent(0) == m);
	assert(linalg::is_symmetric(lhs));
	assert(linalg::is_positive_definite(lhs));
	linalg::cholesky(lhs, rhs);
	assert(_phi.numElements() == rhs.numElements() + 1);
	if (_phi.numElements() > 1) {
		_phi(0, 0, 0) = 0;
	}
	std::copy_n(rhs.data(), rhs.numElements(), _phi.data() + 1);
}

template <class T>
void
arma::generator::AR_model<T>::determine_coefficients_iteratively() {
	using blitz::all;
	using blitz::isfinite;
	using blitz::sum;
	using blitz::RectDomain;
	const Shape3D _0(0, 0, 0);
	Array3D<T> r(this->_acf / this->_acf(0, 0, 0));
	const Shape3D order = this->order();
	Array3D<T> phi0(order), phi1(order);
	phi0 = 0;
	phi1 = 0;
	const int max_order = order(0);
	//			phi0(0, 0, 0) = r(0, 0, 0);
	for (int p = 1; p < max_order; ++p) {
		const Shape3D order(p + 1, p + 1, p + 1);
		/// In three dimensions there are many "last" coefficients. We
		/// collect all their indices into a container to iterate over
		/// them.
		std::vector<Shape3D> indices;
		// for (int i = 0; i < p; ++i) indices.emplace_back(i, p, p);
		// for (int i = 0; i < p; ++i) indices.emplace_back(p, i, p);
		// for (int i = 0; i < p; ++i) indices.emplace_back(p, p, i);
		indices.emplace_back(p, p, p);
		/// Compute coefficients on all three borders.
		for (const Shape3D& idx : indices) {
			const RectDomain<3> sub1(_0, idx), rsub1(idx, _0);
			const T sum1 = sum(phi0(sub1) * r(rsub1));
			const T sum2 = sum(phi0(sub1) * r(sub1));
			phi0(idx) = (r(idx) - sum1) / (T(1) - sum2);
		}
		phi1 = phi0;
		/// Compute all other coefficients.
		{
			using namespace blitz::tensor;
			const Shape3D idx(p, p, p);
			const RectDomain<3> sub(_0, idx), rsub(idx, _0);
			phi1(sub) = phi0(sub) - phi1(p, p, p) * phi0(rsub);
		}
		phi0 = phi1;

		/// Validate white noise variance.
		const T var_wn = white_noise_variance(phi1);
		if (!std::isfinite(var_wn)) {
			std::cerr << "bad white noise variance = " << var_wn << std::endl;
			#ifndef NDEBUG
			std::clog << "Indices: \n";
			std::copy(indices.begin(), indices.end(),
					  std::ostream_iterator<Shape3D>(std::clog, "\n"));
			std::clog << std::endl;
			RectDomain<3> subdomain(_0, order - 1);
			std::clog << "phi1 = \n" << phi1(subdomain) << std::endl;
			#endif
			throw std::runtime_error("bad white noise variance");
		}
		#ifndef NDEBUG
		/// Print solver state.
		std::clog << __func__ << ':' << "Iteration=" << p
				  << ", var_wn=" << var_wn << std::endl;
		#endif

		if (!all(isfinite(phi1))) {
			std::cerr << "bad coefficients = \n" << phi1 << std::endl;
			throw std::runtime_error("bad AR model coefficients");
		}
	}
}

template <class T>
void
arma::generator::AR_model<T>::read(std::istream& in) {
	sys::parameter_map params({
		{"least_squares", sys::make_param(this->_doleastsquares)},
		{"partition", sys::make_param(this->_partition, validate_shape<int,3>)},
	}, true);
	params.insert(this->parameters());
	in >> params;
	this->_phi.resize(this->order());
}

template <class T>
void
arma::generator::AR_model<T>::write(std::ostream& out) const {
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
