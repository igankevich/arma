#ifndef AUTOREG_HH
#define AUTOREG_HH

#include <algorithm>  // for min, any_of, copy_n, for_each, generate
#include <cassert>    // for assert
#include <chrono>     // for duration, steady_clock, steady_clock...
#include <cmath>      // for isnan
#include <cstdlib>    // for abs
#include <functional> // for bind
#include <iostream>   // for operator<<, endl
#include <fstream>    // for ofstream
#include <random>     // for mt19937, normal_distribution
#include <stdexcept>  // for runtime_error
#include <complex>

#include <blitz/array.h> // for Array, Range, shape, any

#include "types.hh"  // for size3, ACF, AR_coefs, Zeta, Array2D
#include "voodoo.hh" // for generate_AC_matrix
#include "linalg.hh"
#include "ma_model.hh"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_poly.h>

/// @file
/// File with subroutines for AR model, Yule-Walker equations
/// and some others.

/// Domain-specific classes and functions.
namespace autoreg {

	template <class T>
	T
	white_noise_variance(const AR_coefs<T>& ar_coefs, const ACF<T>& acf) {
		using blitz::Range;
		const Range ar_range_1(0, ar_coefs.extent(0) - 1);
		const Range ar_range_2(0, ar_coefs.extent(1) - 1);
		const Range ar_range_3(0, ar_coefs.extent(2) - 1);
		return acf(0, 0, 0) -
		       blitz::sum(ar_coefs * acf(ar_range_1, ar_range_2, ar_range_3));
	}

	template <class T>
	T
	ACF_variance(const ACF<T>& acf) {
		return acf(0, 0, 0);
	}

	template <class T>
	void
	check_stationarity(AR_coefs<T>& phi_in) {
		/// Find roots of the polynomial
		/// \f$P_n(\Phi)=1-\Phi_1 x-\Phi_2 x^2 - ... -\Phi_n x^n\f$.
		AR_coefs<double> phi(phi_in.shape());
		phi = -phi_in;
		phi(0) = 1;
		Array1D<std::complex<double>> result(phi.size());
		gsl_poly_complex_workspace* w =
		    gsl_poly_complex_workspace_alloc(phi.size());
		int ret = gsl_poly_complex_solve(phi.data(), phi.size(), w,
		                                 (gsl_complex_packed_ptr)result.data());
		gsl_poly_complex_workspace_free(w);
		if (ret != GSL_SUCCESS) {
			std::clog << "GSL error: " << gsl_strerror(ret) << '.' << std::endl;
			throw std::runtime_error("Can not find roots of the polynomial to "
			                         "verify AR model stationarity.");
		}
		/// Check if some roots do not lie outside unit circle.
		size_t num_bad_roots = 0;
		for (size_t i = 0; i < result.size(); ++i) {
			const double val = std::abs(result(i));
			/// Some AR coefficients are close to nought and polynomial
			/// solver can produce noughts due to limited numerical
			/// precision. So we filter val=0 as well.
			if (!(val > 1.0 || val == 0)) {
				++num_bad_roots;
				std::clog << "Root #" << i << '=' << result(i) << std::endl;
			}
		}
		if (num_bad_roots > 0) {
			std::clog << "No. of bad roots = " << num_bad_roots << std::endl;
			throw std::runtime_error(
			    "AR process is not stationary: some roots lie "
			    "inside unit circle or on its borderline.");
		}
	}

	template <class T>
	AR_coefs<T>
	compute_AR_coefs(const ACF<T>& acf, const size3& ar_order,
	                 bool do_least_squares) {

		if (ar_order(0) > acf.extent(0) || ar_order(1) > acf.extent(1) ||
		    ar_order(2) > acf.extent(2)) {
			std::clog << "AR model order is larger than ACF size:\n\tAR model "
			             "order = "
			          << ar_order << "\n\tACF size = " << acf.shape()
			          << std::endl;
			throw std::runtime_error("bad AR model order");
		}

		using blitz::Range;
		using blitz::toEnd;
		// normalise ACF to prevent big numbers when multiplying matrices
		ACF<T> acf_norm(acf.shape());
		acf_norm = acf / acf(0, 0, 0);
		std::function<Array2D<T>()> generator;
		if (do_least_squares) {
			generator = AC_matrix_generator_LS<T>(acf_norm, ar_order);
		} else {
			generator = AC_matrix_generator<T>(acf_norm, ar_order);
		}
		Array2D<T> acm = generator();
		{
			std::ofstream out("acm");
			out << acm;
		}
		const int m = acm.rows() - 1;

		/**
		eliminate the first equation and move the first column of the remaining
		matrix to the right-hand side of the system
		*/
		Array1D<T> rhs(m);
		rhs = acm(Range(1, toEnd), 0);
		//{ std::ofstream out("rhs"); out << rhs; }

		// lhs is the autocovariance matrix without first
		// column and row
		Array2D<T> lhs(blitz::shape(m, m));
		lhs = acm(Range(1, toEnd), Range(1, toEnd));
		//{ std::ofstream out("lhs"); out << lhs; }

		assert(lhs.extent(0) == m);
		assert(lhs.extent(1) == m);
		assert(rhs.extent(0) == m);
		assert(linalg::is_symmetric(lhs));
		assert(linalg::is_positive_definite(lhs));
		linalg::cholesky(lhs, rhs);
		AR_coefs<T> phi(ar_order);
		assert(phi.numElements() == rhs.numElements() + 1);
		phi(0, 0, 0) = 0;
		std::copy_n(rhs.data(), rhs.numElements(), phi.data() + 1);
		{
			std::ofstream out("ar_coefs");
			out << phi;
		}
		return phi;
	}

	template <class T>
	bool
	isnan(T rhs) noexcept {
		return std::isnan(rhs);
	}

	template <class T>
	bool
	isinf(T rhs) noexcept {
		return std::isinf(rhs);
	}

	/**
	Generate white noise via Mersenne Twister algorithm. Convert to normal
	distribution via Box---Muller transform.
	*/
	template <class T>
	Zeta<T>
	generate_white_noise(const size3& size, T variance) {
		if (variance < T(0)) {
			throw std::runtime_error("variance is less than zero");
		}

		// initialise generator
		std::mt19937 generator;
#if !defined(DISABLE_RANDOM_SEED)
		generator.seed(
		    std::chrono::steady_clock::now().time_since_epoch().count());
#endif
		std::normal_distribution<T> normal(T(0), std::sqrt(variance));

		// generate and check
		Zeta<T> eps(size);
		std::generate(std::begin(eps), std::end(eps),
		              std::bind(normal, generator));
		if (std::any_of(std::begin(eps), std::end(eps), &::autoreg::isnan<T>)) {
			throw std::runtime_error(
			    "white noise generator produced some NaNs");
		}
		if (std::any_of(std::begin(eps), std::end(eps), &::autoreg::isinf<T>)) {
			throw std::runtime_error(
			    "white noise generator produced infinite numbers");
		}
		return eps;
	}

	/**
	Generate wavy surface realisation.
	*/
	template <class T>
	void
	generate_zeta(AR_coefs<T>& phi, Zeta<T>& zeta) {
		const size3 fsize = phi.shape();
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
								sum += phi(k, i, j) * zeta(t - k, x - i, y - j);
					zeta(t, x, y) += sum;
				}
			}
		}
	}

	template <class T, int N>
	T
	mean(blitz::Array<T, N> rhs) {
		assert(rhs.numElements() > 0);
		return blitz::sum(rhs) / rhs.numElements();
	}

	template <class T, int N>
	T
	variance(blitz::Array<T, N> rhs) {
		assert(rhs.numElements() > 1);
		const T m = mean(rhs);
		return blitz::sum(blitz::pow2(rhs - m)) / (rhs.numElements() - 1);
	}
}

#endif // AUTOREG_HH
