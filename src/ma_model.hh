#ifndef MA_MODEL_HH
#define MA_MODEL_HH

#include <cassert>   // for assert
#include <algorithm> // for copy_n

#include "types.hh" // for size3, ACF, AR_coefs, Zeta, Array2D
#include "linalg.hh"

namespace autoreg {

	template <class T>
	struct Moving_average_model {

		Moving_average_model(ACF<T> acf, size3 order)
		    : _acf(acf), _order(order), _theta(_order) {}

		T
		white_noise_variance(const Array3D<T>& theta) {
			return _acf(0, 0, 0) / (T(1) + blitz::sum(blitz::pow2(theta)));
		}

		/**
		Compute white noise variance via the following formula.
		\f[
		    \sigma_\alpha^2 = \frac{\gamma_0}{
		        1
		        +
		        \sum\limits_{i=0}^{n_1}
		        \sum\limits_{i=0}^{n_2}
		        \sum\limits_{k=0}^{n_3}
		        \theta_{i,j,k}^2
		    }
		\f]
		*/
		T
		white_noise_variance() {
			return white_noise_variance(_theta);
		}

		Zeta<T> operator()(Zeta<T> eps) {
			Zeta<T> zeta(eps.shape());
			const size3 fsize = _theta.shape();
			const size3 zsize = zeta.shape();
			const int t1 = zsize[0];
			const int x1 = zsize[1];
			const int y1 = zsize[2];
			for (int t = 0; t < t1; t++) {
				for (int x = 0; x < x1; x++) {
					for (int y = 0; y < y1; y++) {
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
			return zeta;
		}

		/**
		Solve nonlinear system with fixed-point iteration algorithm to find
		moving-average coefficients \f$\theta\f$.

		\param max_iterations Maximal no. of iterations.
		\param eps
		\parblock
		    Maximal difference between values of white
		    noise variance in successive iterations.
		\endparblock
		\param min_var_wn
		\parblock
		    Maximal value of white noise variance considered to be
		    nought.
		\endparblock
		*/
		void
		determine_coefficients(int max_iterations, T eps, T min_var_wn) {
			Array3D<T> theta(_order);
			theta = 0;
			const int order_t = _order(0);
			const int order_x = _order(1);
			const int order_y = _order(2);
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
							T sum = -_acf(i, j, k) / var_wn;
							for (int l = i; l < order_t; ++l) {
								for (int m = j; m < order_x; ++m) {
									for (int n = k; n < order_y; ++n) {
										sum += theta(l, m, n) *
										       theta(l - i, m - j, n - k);
									}
								}
							}
							theta(i, j, k) = sum;
						}
					}
				}
				/// 3. Zero out \f$\theta_0\f$.
				theta(0, 0, 0) = 0;
				/// 4. Validate coefficients.
				if (!blitz::all(blitz::isfinite(theta))) {
					std::clog << __FILE__ << ':' << __LINE__ << ':' << __func__
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
					std::clog << __FILE__ << ':' << __LINE__ << ':' << __func__
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

		void
		validate() {
			validate_process(_theta);
		}

		void
		determine_coefficients_newton_raphson(int max_iterations, T eps,
		                                      T min_var_wn) {
			using blitz::RectDomain;
			using blitz::TinyVector;
			using blitz::sum;
			using blitz::all;
			using blitz::isfinite;
			const int n = blitz::product(_order);
			Array3D<T> theta(_order), tau(_order), f(_order);
			Array2D<T> tau_matrix(n, n);
			theta = 0;
			tau = 0;
			const int order_t = _order(0);
			const int order_x = _order(1);
			const int order_y = _order(2);
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
				{
					using namespace blitz::tensor;
					RectDomain<3> sub1(size3(0, 0, 0),
					                   _order - size3(i, j, k) - 1);
					RectDomain<3> sub2(size3(i, j, k), _order - 1);
					f = sum(tau(sub1) * tau(sub2)) - _acf;
				}
				{
					using namespace blitz::tensor;
					tau_matrix = 0;
					for (int i=0; i<n; ++i) {
						for (int j=0; j<n-i; ++j) {
							tau_matrix(i,j) = tau.data()[i + j];
						}
					}
					for (int i=0; i<n; ++i) {
						for (int j=i; j<n; ++j) {
							tau_matrix(i,j) += tau.data()[j - i];
						}
					}
				}
				linalg::invert(tau_matrix);
				tau -= linalg::operator*(tau_matrix, f);
				theta = -tau / tau(0, 0, 0);
				/// 4. Validate coefficients.
				if (!all(isfinite(theta))) {
					std::clog << __FILE__ << ':' << __LINE__ << ':' << __func__
					          << ": bad coefficients = \n" << theta
					          << std::endl;
					throw std::runtime_error("bad MA model coefficients");
				}
				/// 5. Compute white noise variance by calling
				/// \link Moving_average_model::white_noise_variance \endlink.
				old_var_wn = var_wn;
				var_wn = white_noise_variance(theta);
				tau(0,0,0) = std::sqrt(var_wn);
				/// 6. Validate white noise variance.
				if (var_wn <= min_var_wn || !std::isfinite(var_wn)) {
					std::clog << __FILE__ << ':' << __LINE__ << ':' << __func__
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

	private:
		ACF<T> _acf;
		size3 _order;
		AR_coefs<T> _theta;
	};
}

#endif // MA_MODEL_HH
