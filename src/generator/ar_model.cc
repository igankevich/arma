#include "ar_model.hh"

#include "voodoo.hh"
#include "linalg.hh"
#include "params.hh"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>

template <class T>
T
arma::generator::AR_model<T>::white_noise_variance(Array3D<T> phi) const {
	blitz::RectDomain<3> subdomain(Shape3D(0, 0, 0), phi.shape() - 1);
	return this->_acf(0,0,0) - blitz::sum(phi * this->_acf(subdomain));
}

template <class T>
void
arma::generator::AR_model<T>::validate() const {
	using blitz::all;
	if (!all(this->_phi.shape() == this->order())) {
		throw std::runtime_error("bad shape");
	}
	validate_process(this->_phi);
}

template <class T>
void
arma::generator::AR_model<T>::generate_surface(
	Array3D<T>& zeta,
	Array3D<T>& eps,
	const Domain3D& subdomain
) {
	if (std::addressof(zeta) != std::addressof(eps)) {
		zeta(subdomain) = eps(subdomain);
	}
	const Shape3D fsize = _phi.shape();
	const Shape3D& lbound = subdomain.lbound();
	const Shape3D& ubound = subdomain.ubound();
	const int t0 = lbound(0);
	const int x0 = lbound(1);
	const int y0 = lbound(2);
	const int t1 = ubound(0);
	const int x1 = ubound(1);
	const int y1 = ubound(2);
	for (int t = t0; t <= t1; t++) {
		for (int x = x0; x <= x1; x++) {
			for (int y = y0; y <= y1; y++) {
				const int m1 = std::min(t + 1, fsize[0]);
				const int m2 = std::min(x + 1, fsize[1]);
				const int m3 = std::min(y + 1, fsize[2]);
				T sum = 0;
				for (int k = 0; k < m1; k++)
					for (int i = 0; i < m2; i++)
						for (int j = 0; j < m3; j++)
							sum += _phi(k, i, j) *
								   zeta(t - k, x - i, y - j);
				zeta(t, x, y) += sum;
			}
		}
	}
}

template <class T>
void
arma::generator::AR_model<T>::determine_coefficients_old(bool do_least_squares) {
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
		<< ",acf.shape=" << this->_acf.shape();
}

template class arma::generator::AR_model<ARMA_REAL_TYPE>;
