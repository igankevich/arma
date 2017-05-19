#include "ma_model.hh"

#include "linalg.hh"
#include "voodoo.hh"
#include "validators.hh"
#include "params.hh"
#include "acf.hh"

#include <cassert>
#include <algorithm>

template <class T>
T
arma::Moving_average_model<T>::white_noise_variance(
	const Array3D<T>& theta
) const {
	using blitz::sum;
	using blitz::pow2;
	return _acf(0, 0, 0) / (T(1) + sum(pow2(theta)));
}

template <class T>
void
arma::Moving_average_model<T>::operator()(
	Array3D<T>& zeta,
	Array3D<T>& eps,
	const Domain3D& subdomain
) {
	const Shape3D fsize = _theta.shape();
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
							sum += _theta(k, i, j) *
								   eps(t - k, x - i, y - j);
				zeta(t, x, y) = eps(t, x, y) - sum;
			}
		}
	}
}

template <class T>
std::istream&
arma::operator>>(std::istream& in, Moving_average_model<T>& rhs) {
	ACF_wrapper<T> acf_wrapper(rhs._acf);
	Shape3D order(0,0,0);
	sys::parameter_map params({
		{"order", sys::make_param(order)},
		{"acf", sys::make_param(acf_wrapper)},
		{"algorithm", sys::make_param(rhs._algo)},
		{"max_iterations", sys::make_param(rhs._maxiter)},
		{"eps", sys::make_param(rhs._eps)},
		{"min_var_wn", sys::make_param(rhs._minvarwn)},
	}, true);
	in >> params;
	validate_shape(order, "ma_model.order");
	validate_shape(rhs._acf.shape(), "ma_model.acf.shape");
	validate_positive(rhs._maxiter, "ma_model.algo.max_iterations");
	validate_positive(rhs._eps, "ma_model.algo.eps");
	validate_positive(rhs._minvarwn, "ma_model.algo.min_var_wn");
	rhs._theta.resize(order);
	return in;
}

template <class T>
std::ostream&
arma::operator<<(std::ostream& out, const Moving_average_model<T>& rhs) {
	return out << "order=" << rhs.order()
		<< ",acf.shape=" << rhs._acf.shape()
		<< ",algorithm=" << rhs._algo;
}

template <class T>
void
arma::Moving_average_model<T>::determine_coefficients() {
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
arma::Moving_average_model<T>::fixed_point_iteration(
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
	T var_wn = _acf(0, 0, 0);
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
						-_acf(i, j, k) / var_wn +
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
		/// \link Moving_average_model::white_noise_variance \endlink.
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
arma::Moving_average_model<T>::newton_raphson(
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
	T var_wn = _acf(0, 0, 0);
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
						sum(tau(sub1) * tau(sub2)) - _acf(i, j, k);
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
		/// \link Moving_average_model::white_noise_variance \endlink.
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
arma::Moving_average_model<T>::recompute_acf(
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
				_acf(i, j, k) =
					sum_phi_1 * acf_orig(i, j, k) + sum_phi_2;
			}
		}
	}
}

template class arma::Moving_average_model<ARMA_REAL_TYPE>;

template std::istream&
arma::operator>>(
	std::istream& in,
	Moving_average_model<ARMA_REAL_TYPE>& rhs
);

template std::ostream&
arma::operator<<(
	std::ostream& out,
	const Moving_average_model<ARMA_REAL_TYPE>& rhs
);
