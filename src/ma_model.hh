#ifndef MA_MODEL_HH
#define MA_MODEL_HH

#include <cassert>   // for assert
#include <algorithm> // for copy_n

#include "types.hh" // for size3, ACF, AR_coefs, Zeta, Array2D

namespace autoreg {

	template <class T>
	struct Moving_average_model {

		Moving_average_model(ACF<T> acf, size3 order)
		    : _acf(acf), _order(order), _theta(_order) {}

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
		white_noise_variance(const Array3D<T>& theta) {
			return _acf(0, 0, 0) / (T(1) + blitz::sum(blitz::pow2(theta)));
		}

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
						zeta(t, x, y) = sum;
					}
				}
			}
			return zeta;
		}

		/// Solve nonlinear system with fixed-point iteration algorithm to find
		/// moving-average coefficients \f$\theta\f$.
		void
		determine_coefficients(int max_iterations, T eps) {
			Array3D<T> theta(_order);
			theta = 0;
			const int order_t = _order(0);
			const int order_x = _order(1);
			const int order_y = _order(2);
			for (int it = 0; it < max_iterations; ++it) {
				/// 1. Compute white noise variance.
				/// \see white_noise_variance
				const T var_wn = white_noise_variance(theta);
				/// 2. Validate white noise variance.
				if (var_wn == T(0) || !std::isfinite(var_wn)) {
					std::clog << __FILE__ << ':' << __LINE__ << ':' << __func__
					          << ": bad white noise variance = " << var_wn
					          << std::endl;
					throw std::runtime_error("bad white noise variance");
				}
				/// 3. Update coefficients from back to front.
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
				/// 4. Zero out \f$\theta_0\f$.
				theta(0, 0, 0) = 0;
				/// 5. Validate coefficients.
				if (!blitz::all(blitz::isfinite(theta))) {
					std::clog << __FILE__ << ':' << __LINE__ << ':' << __func__
					          << ": bad coefficients = \n" << theta
					          << std::endl;
					throw std::runtime_error("bad MA model coefficients");
				}
#ifndef NDEBUG
				/// 5. Print solver state.
				std::clog << "Iteration=" << it << ", var_wn=" << var_wn
				          << std::endl;
#endif
			}
			_theta = theta;
		}

		/// TODO
		void
		validate() {}

	private:
		ACF<T> _acf;
		size3 _order;
		AR_coefs<T> _theta;
	};
}

#endif // MA_MODEL_HH
