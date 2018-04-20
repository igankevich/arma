#include "ma_coefficient_solver.hh"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "blitz.hh"
#include "params.hh"
#include "validators.hh"

template <class T>
arma::generator::MA_coefficient_solver<T>
::MA_coefficient_solver(Array3D<T> acf, const Shape3D& order):
_acf(acf),
_order(order) {
	using blitz::all;
	if (!all(this->_order <= this->_acf.shape())) {
		throw std::invalid_argument("bad order");
	}
}

template <class T>
arma::Array3D<T>
arma::generator::MA_coefficient_solver<T>
::solve() {
	return this->solve_fixed_point_iteration();
}

template <class T>
arma::Array3D<T>
arma::generator::MA_coefficient_solver<T>
::solve_fixed_point_iteration() {
	using blitz::RectDomain;
	using blitz::abs;
	using blitz::all;
	using blitz::isfinite;
	using blitz::max;
	using blitz::sum;
	using std::abs;
	const int max_iterations = this->_maxiterations;
	const T min_var_wn = this->_minvarwn;
	const T eps = this->_epsvarwn;
	const T max_residual = this->_maxresidual;
	const Shape3D order = this->_order;
	Array3D<T> theta(order);
	theta = 0;
	const int ni = order(0);
	const int nj = order(1);
	const int nk = order(2);
	/// 1. Precompute white noise variance for the first iteration.
	T var_wn = this->_acf(0,0,0);
	T old_var_wn = 0;
	T residual = 0;
	int it = 0;
	std::ofstream out("theta_all");
	do {
		/**
		   2. Update coefficients from back to front using the
		   following formula (adapted from G. Box and G. Jenkins (1970)
		   "Time Series Analysis: Forecasting and Control", pp. 226--227).
		   \f[
		    \theta_{i,j,k} = -\frac{\gamma_0}{\sigma_\alpha^2} +
		        \sum\limits_{l=i}^{n_1}
		        \sum\limits_{m=j}^{n_2}
		        \sum\limits_{n=k}^{n_3}
		        \theta_{l,m,n} \theta_{l-i,m-j,n-k}
		   \f]
		   Here \f$\theta_{0,0,0} \equiv 0\f$.
		 */
		theta(0,0,0) = 0;
		for (int i = ni - 1; i >= 0; --i) {
			for (int j = nj - 1; j >= 0; --j) {
				for (int k = nk - 1; k >= 0; --k) {
					const Shape3D ijk(i,j,k);
					RectDomain<3> sub1(ijk, order - 1);
					RectDomain<3> sub2(Shape3D(0,0,0), order - ijk - 1);
					theta(i,j,k) =
						-this->_acf(i,j,k) / var_wn +
						sum(theta(sub1)*theta(sub2));
				}
			}
		}
		theta(0,0,0) = -1;
		for (T t : theta) {
			out << t << '\n';
		}
		out << "\n\n";
		out << std::flush;
		/// 3. Ensure that coefficients are finite.
		if (!all(isfinite(theta))) {
			std::cerr << __func__
			          << ": bad coefficients"
			          << std::endl;
			throw std::runtime_error("bad MA model coefficients");
		}
		/// 4. Calculate residual. Here \f$\theta_{0,0,0} \equiv -1\f$.
		theta(0,0,0) = -1;
		residual = std::numeric_limits<T>::min();
		for (int i=0; i<ni; ++i) {
			for (int j=0; j<nj; ++j) {
				for (int k=0; k<nk; ++k) {
					const Shape3D ijk(i,j,k);
					RectDomain<3> sub1(ijk, order-1);
					RectDomain<3> sub2(Shape3D(0,0,0), order-ijk-1);
					T new_residual = abs(
						this->_acf(i,j,k) - sum(theta(sub1)*theta(sub2))*var_wn
					                 );
					if (residual < new_residual) {
						residual = new_residual;
					}
				}
			}
		}
		/// 5. Compute white noise variance by calling
		/// \link MA_coefficient_solver::white_noise_variance \endlink.
		old_var_wn = var_wn;
		var_wn = this->white_noise_variance(theta);
		/// 6. Validate white noise variance.
		if (var_wn <= min_var_wn) {
			std::cerr << __func__
			          << ": bad white noise variance = " << var_wn
			          << std::endl;
			throw std::runtime_error("bad white noise variance");
		}
//		#ifndef NDEBUG
		/// 8. Print solver state.
		std::clog << __func__ << ':' << "Iteration=" << it
		          << ",var_wn=" << var_wn
		          << ",resudual=" << residual
		          << ",theta(0,0,0)=" << theta(0,0,0)
		          << std::endl;
//		#endif
		++it;
	} while ((it < max_iterations) &&
	         abs(var_wn - old_var_wn) > eps &&
	         (residual > max_residual));
	std::clog << "Calculated MA model coefficients:"
	          << "\n\tno. of iterations = " << it
	          << "\n\twhite noise variance = " << var_wn
	          << "\n\tmax(theta) = " << max(abs(theta))
	          << "\n\tmax(residual) = " << residual
	          << std::endl;
	return theta;
}

template <class T>
std::istream&
arma::generator::operator>>(std::istream& in, MA_coefficient_solver<T>& rhs) {
	sys::parameter_map params {
		{
			{
				"max_iterations",
				sys::make_param(rhs._maxiterations, validate_positive<int>)
			},
			{
				"eps",
				sys::make_param(rhs._epsvarwn, validate_positive<T>)
			},
			{
				"min_white_noise_variance",
				sys::make_param(rhs._minvarwn, validate_positive<T>)
			},
			{
				"max_residual",
				sys::make_param(rhs._maxresidual, validate_positive<T>)
			},
		},
		true
	};
	return in;
}

template class arma::generator::MA_coefficient_solver<ARMA_REAL_TYPE>;

template std::istream&
arma::generator::operator>>(
	std::istream& in,
	MA_coefficient_solver<ARMA_REAL_TYPE>& rhs
);
