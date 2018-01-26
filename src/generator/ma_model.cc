#include "ma_model.hh"

#include "apmath/convolution.hh"
#include "linalg.hh"
#include "ma_coefficient_solver.hh"
#include "params.hh"
#include "profile.hh"
#include "validators.hh"
#include "voodoo.hh"

#include <algorithm>
#include <cassert>
#include <complex>
#include <random>

template <class T>
T
arma::generator::MA_model<T>
::white_noise_variance(const Array3D<T>& theta) const {
	return MA_white_noise_variance(this->_acf, theta);
}

template <class T>
void
arma::generator::MA_model<T>
::validate() const {
	validate_process(this->_theta);
}

template <class T>
arma::Array3D<T>
arma::generator::MA_model<T>
::do_generate() {
	ARMA_PROFILE_START(generate_white_noise);
	Array3D<T> eps = this->generate_white_noise();
	ARMA_PROFILE_END(generate_white_noise);
	Array3D<T> zeta(this->grid().num_points());
	generate_surface(zeta, eps, zeta.domain());
	return zeta;
}

template <class T>
void
arma::generator::MA_model<T>
::generate_surface(
	Array3D<T>& zeta,
	Array3D<T>& eps,
	const Domain3D& subdomain
) {
	using blitz::real;
	typedef std::complex<T> C;
	typedef apmath::Convolution<C,3> convolution_type;
	Array3D<C> signal(eps.shape());
	signal = eps;
	Array3D<C> kernel(this->_theta.shape());
	kernel = this->_theta;
	kernel(0,0,0) = -1;
	kernel = -kernel;
	convolution_type conv(signal, kernel);
	zeta = real(conv.convolve(signal, kernel));
}

template <class T>
void
arma::generator::MA_model<T>
::read(std::istream& in) {
	typedef typename Basic_model<T>::grid_type grid_type;
	sys::parameter_map params({
	                              {"algorithm", sys::make_param(this->_algo)},
							  }, true);
	params.insert(this->parameters());
	in >> params;
	// resize output grid to match ACF delta size
	this->_outgrid = grid_type(
		this->_outgrid.num_points(),
		this->_acf.grid().delta() * this->_outgrid.num_patches()
	                 );
	this->_theta.resize(this->order());
}

template <class T>
void
arma::generator::MA_model<T>
::write(std::ostream& out) const {
	out << "grid=" << this->grid()
	    << ",order=" << this->order()
	    << ",output=" << this->_oflags
	    << ",acf.shape=" << this->_acf.shape()
	    << ",algorithm=" << this->_algo;
}

template <class T>
void
arma::generator::MA_model<T>
::determine_coefficients() {
	switch (this->_algo) {
	case MA_algorithm::Fixed_point_iteration:
		this->fixed_point_iteration();
		break;
	default:
		throw std::runtime_error("only fixed_point_iteration is supported");
		break;
	}
}

template <class T>
void
arma::generator::MA_model<T>
::recompute_acf(
	Array3D<T> acf_orig,
	Array3D<T> phi
) {
	using blitz::sum;
	using blitz::pow2;
	using blitz::RectDomain;
	using blitz::abs;
	const Shape3D _0(0, 0, 0);
	const Shape3D ar_order = phi.shape();
	const Shape3D& order = this->order();
	const T sum_phi_1 = sum(pow2(phi));
	const int ma_order_t = order(0);
	const int ma_order_x = order(1);
	const int ma_order_y = order(2);
	const int ar_order_t = ar_order(0);
	const int ar_order_x = ar_order(1);
	const int ar_order_y = ar_order(2);
	for (int i = 0; i < ma_order_t; ++i) {
		for (int j = 0; j < ma_order_x; ++j) {
			for (int k = 0; k < ma_order_y; ++k) {
				const Shape3D ijk(i, j, k);
				T sum_phi_2 = 0;
				for (int l = 0; l < ar_order_t; ++l) {
					for (int m = 0; m < ar_order_x; ++m) {
						for (int n = 0; n < ar_order_y; ++n) {
							const Shape3D lmn(l, m, n);
							const Shape3D ijk_plus_lmn(ijk + lmn);
							const Shape3D ijk_minus_lmn(abs(ijk - lmn));
							RectDomain<3> sub1(_0, ar_order - lmn - 1),
							sub2(lmn, ar_order - 1);
							sum_phi_2 += sum(phi(sub1) * phi(sub2)) *
							             (acf_orig(ijk_plus_lmn) +
							              acf_orig(ijk_minus_lmn));
						}
					}
				}
				this->_acf(i, j, k) =
					sum_phi_1 * acf_orig(i, j, k) + sum_phi_2;
			}
		}
	}
}

template <class T>
void
arma::generator::MA_model<T>
::fixed_point_iteration() {
	MA_coefficient_solver<T> solver(this->_acf, this->_order);
	this->_theta.reference(solver.solve());
	this->_varwn = solver.white_noise_variance(this->_theta);
}

template class arma::generator::MA_model<ARMA_REAL_TYPE>;
