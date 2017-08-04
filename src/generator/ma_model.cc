#include "ma_model.hh"

#include "linalg.hh"
#include "voodoo.hh"
#include "validators.hh"
#include "params.hh"
#include "apmath/convolution.hh"
#include "profile.hh"

#include <cassert>
#include <algorithm>
#include <random>
#include <complex>

template <class T>
T
arma::generator::MA_model<T>::white_noise_variance(
	const Array3D<T>& theta
) const {
	using blitz::sum;
	using blitz::pow2;
	return this->_acf(0,0,0) / (T(1) + sum(pow2(theta)));
}

template <class T>
void
arma::generator::MA_model<T>::validate() const {
	validate_process(this->_theta);
}

template <class T>
arma::Array3D<T>
arma::generator::MA_model<T>::do_generate() {
	ARMA_PROFILE_START(generate_white_noise);
	Array3D<T> eps = this->generate_white_noise();
	ARMA_PROFILE_END(generate_white_noise);
	Array3D<T> zeta(this->grid().num_points());
	generate_surface(zeta, eps, zeta.domain());
	return zeta;
}

template <class T>
void
arma::generator::MA_model<T>::generate_surface(
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
arma::generator::MA_model<T>::read(std::istream& in) {
	sys::parameter_map params({
		{"algorithm", sys::make_param(this->_algo)},
		{"max_iterations", sys::make_param(this->_maxiter, validate_positive<int>)},
		{"eps", sys::make_param(this->_eps, validate_positive<T>)},
		{"min_var_wn", sys::make_param(this->_minvarwn, validate_positive<T>)},
	}, true);
	params.insert(this->parameters());
	in >> params;
	this->_theta.resize(this->order());
}

template <class T>
void
arma::generator::MA_model<T>::write(std::ostream& out) const {
	out << "grid=" << this->grid()
		<< ",order=" << this->order()
		<< ",output=" << this->_oflags
		<< ",acf.shape=" << this->_acf.shape()
		<< ",algorithm=" << this->_algo;
}

template <class T>
void
arma::generator::MA_model<T>::determine_coefficients() {
	switch (_algo) {
		case MA_algorithm::Fixed_point_iteration:
			fixed_point_iteration(_maxiter, _eps, _minvarwn);
			break;
		case MA_algorithm::Newton_Raphson:
			newton_raphson(_maxiter, _eps, _minvarwn);
			break;
	}
}

template <class T>
void
arma::generator::MA_model<T>::fixed_point_iteration(
	int max_iterations,
	T eps,
	T min_var_wn
) {
	using blitz::RectDomain;
	const Shape3D order = this->order();
	Array3D<T> theta(order);
	theta = 0;
	const int order_t = order(0);
	const int order_x = order(1);
	const int order_y = order(2);
	/// 1. Precompute white noise variance for the first iteration.
	T var_wn = this->_acf(0, 0, 0);
	T old_var_wn = 0;
	int it = 0;
	do {
		/**
		2. Update coefficients from back to front using the
		following formula (adapted from G. Box and G. Jenkins (1970)
		"Time Series Analysis: Forecasting and Control", pp. 226--227).
		\f[
			\theta_{i,j,k} = -\frac{\gamma_0}{\sigma_\alpha^2}
				+
				\sum\limits_{l=i}^{n_1}
				\sum\limits_{m=j}^{n_2}
				\sum\limits_{n=k}^{n_3}
				\theta_{l,m,n} \theta_{l-i,m-j,n-k}
		\f]
		Here \f$\theta_{0,0,0} \equiv 0\f$.
		*/
		for (int i = order_t - 1; i >= 0; --i) {
			for (int j = order_x - 1; j >= 0; --j) {
				for (int k = order_y - 1; k >= 0; --k) {
					RectDomain<3> sub1(Shape3D(i, j, k), order - 1);
					RectDomain<3> sub2(Shape3D(0, 0, 0),
									   order - Shape3D(i, j, k) - 1);
					theta(i, j, k) =
						-this->_acf(i, j, k) / var_wn +
						blitz::sum(theta(sub1) * theta(sub2));
				}
			}
		}
		/// 3. Zero out \f$\theta_0\f$.
		theta(0, 0, 0) = 0;
		/// 4. Validate coefficients.
		if (!blitz::all(blitz::isfinite(theta))) {
			std::cerr << __func__
					  << ": bad coefficients = \n" << theta
					  << std::endl;
			throw std::runtime_error("bad MA model coefficients");
		}
		/// 5. Compute white noise variance by calling
		/// \link MA_model::white_noise_variance \endlink.
		old_var_wn = var_wn;
		var_wn = white_noise_variance(theta);
		/// 6. Validate white noise variance.
		if (var_wn <= min_var_wn || !std::isfinite(var_wn)) {
			std::cerr << __func__
					  << ": bad white noise variance = " << var_wn
					  << std::endl;
			throw std::runtime_error("bad white noise variance");
		}
		#ifndef NDEBUG
		/// 7. Print solver state.
		std::clog << __func__ << ':' << "Iteration=" << it
				  << ", var_wn=" << var_wn << std::endl;
		#endif
		++it;
	} while (it < max_iterations &&
			 std::abs(var_wn - old_var_wn) > eps);
	_theta = theta;
}

template <class T>
void
arma::generator::MA_model<T>::newton_raphson(
	int max_iterations,
	T eps,
	T min_var_wn
) {
	using blitz::RectDomain;
	using blitz::TinyVector;
	using blitz::sum;
	using blitz::all;
	using blitz::isfinite;
	const Shape3D& order = this->order();
	const int n = blitz::product(order);
	Array3D<T> theta(order), tau(order), f(order);
	Array2D<T> tau_matrix(n, n);
	theta = 0;
	tau = 0;
	const int order_t = order(0);
	const int order_x = order(1);
	const int order_y = order(2);
	/// 1. Precompute white noise variance for the first iteration.
	T var_wn = this->_acf(0, 0, 0);
	tau(0, 0, 0) = std::sqrt(var_wn);
	T old_var_wn = 0;
	int it = 0;
	do {
		/**
		2. Update coefficients using the following formula (adapted from
		G. Box and G. Jenkins (1970) "Time Series Analysis: Forecasting
		and Control", p. 227).
		\f[
			\theta_{i,j,k} = -\frac{\tau_{i,j,k}}{\tau{0,0,0}},
			f_{i,j,k} =
				\sum\limits_{l=0}^{n_1-i}
				\sum\limits_{m=0}^{n_2-j}
				\sum\limits_{n=0}^{n_3-k}
				\tau_{l,m,n} \tau_{l+i,m+j,n+k} - r_{i,j,k}.
		\f]
		Here \f$\tau_{0,0,0}^2 = \sigma_\alpha^2\f$.
		*/
		for (int i = 0; i < order_t; ++i) {
			for (int j = 0; j < order_x; ++j) {
				for (int k = 0; k < order_y; ++k) {
					RectDomain<3> sub1(Shape3D(0, 0, 0),
									   order - Shape3D(i, j, k) - 1);
					RectDomain<3> sub2(Shape3D(i, j, k), order - 1);
					f(i, j, k) =
						sum(tau(sub1) * tau(sub2)) - this->_acf(i, j, k);
				}
			}
		}
		{
			Tau_matrix_generator<T> gen(tau);
			tau_matrix = gen();
			//					tau_matrix = 0;
			//					for (int i = 0; i < n; ++i) {
			//						for (int j = 0; j < n - i; ++j) {
			//							tau_matrix(i, j) = tau.data()[i
			//+
			// j];
			//						}
			//					}
			//					for (int i = 0; i < n; ++i) {
			//						for (int j = i; j < n; ++j) {
			//							tau_matrix(i, j) += tau.data()[j
			//-
			// i];
			//						}
			//					}
		}
		linalg::inverse(tau_matrix);
		tau -= linalg::operator*(tau_matrix, f);
		theta = -tau / tau(0, 0, 0);
		/// 3. Zero out \f$\theta_0\f$.
		theta(0, 0, 0) = 0;
		/// 4. Validate coefficients.
		if (!all(isfinite(theta))) {
			std::cerr << __func__
					  << ": bad coefficients = \n" << theta
					  << std::endl;
			throw std::runtime_error("bad MA model coefficients");
		}
		/// 5. Compute white noise variance by calling
		/// \link MA_model::white_noise_variance \endlink.
		old_var_wn = var_wn;
		var_wn = white_noise_variance(theta);
		tau(0, 0, 0) = std::sqrt(var_wn);
		/// 6. Validate white noise variance.
		if (var_wn <= min_var_wn || !std::isfinite(var_wn)) {
			std::cerr << __func__
					  << ": bad white noise variance = " << var_wn
					  << std::endl;
			throw std::runtime_error("bad white noise variance");
		}
		#ifndef NDEBUG
		/// 7. Print solver state.
		std::clog << __func__ << ':' << "Iteration=" << it
				  << ", var_wn=" << var_wn << std::endl;
		#endif
		++it;
	} while (it < max_iterations &&
			 std::abs(var_wn - old_var_wn) > eps);
	_theta = theta;
}

template <class T>
void
arma::generator::MA_model<T>::recompute_acf(
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

template class arma::generator::MA_model<ARMA_REAL_TYPE>;
