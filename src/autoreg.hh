#ifndef AUTOREG_HH
#define AUTOREG_HH

#include <algorithm>  // for any_of, generate
#include <cassert>    // for assert
#include <chrono>     // for duration, steady_clock, steady_clock::...
#include <cmath>      // for isinf, isnan
#include <functional> // for bind
#include <random>     // for mt19937, normal_distribution
#include <stdexcept>  // for runtime_error

#include <blitz/array.h>     // for Array
#include <gsl/gsl_complex.h> // for gsl_complex_packed_ptr
#include <gsl/gsl_errno.h>   // for gsl_strerror, ::GSL_SUCCESS
#include <gsl/gsl_poly.h>    // for gsl_poly_complex_solve, gsl_poly_com...

#include "types.hh" // for Zeta, ACF, size3

/// @file
/// File with auxiliary subroutines.

namespace blitz {

	bool
	isfinite(float rhs) noexcept {
		return std::isfinite(rhs);
	}

	bool
	isfinite(double rhs) noexcept {
		return std::isfinite(rhs);
	}

	BZ_DECLARE_FUNCTION(isfinite);
}

/// Domain-specific classes and functions.
namespace autoreg {

	/// Check AR (MA) process stationarity (invertibility).
	template <class T, int N>
	void
	validate_process(blitz::Array<T, N> _phi) {
		/// 1. Find roots of the polynomial
		/// \f$P_n(\Phi)=1-\Phi_1 x-\Phi_2 x^2 - ... -\Phi_n x^n\f$.
		blitz::Array<double, N> phi(_phi.shape());
		phi = -_phi;
		phi(0) = 1;
		/// 2. Trim leading zero terms.
		size_t real_size = 0;
		while (real_size < phi.numElements() && phi.data()[real_size] != 0.0) {
			++real_size;
		}
		blitz::Array<std::complex<double>, 1> result(real_size);
		gsl_poly_complex_workspace* w =
		    gsl_poly_complex_workspace_alloc(real_size);
		int ret = gsl_poly_complex_solve(phi.data(), real_size, w,
		                                 (gsl_complex_packed_ptr)result.data());
		gsl_poly_complex_workspace_free(w);
		if (ret != GSL_SUCCESS) {
			std::clog << "GSL error: " << gsl_strerror(ret) << '.' << std::endl;
			throw std::runtime_error("Can not find roots of the polynomial to "
			                         "verify AR/MA model stationarity/invertibility.");
		}
		/// 3. Check if some roots do not lie outside unit circle.
		size_t num_bad_roots = 0;
		for (size_t i = 0; i < result.size(); ++i) {
			const double val = std::abs(result(i));
			/// Some AR coefficients are close to nought and polynomial
			/// solver can produce noughts due to limited numerical
			/// precision. So we filter val=0 as well.
			if (!(val > 1.0 || val == 0.0)) {
				++num_bad_roots;
				std::clog << "Root #" << i << '=' << result(i) << std::endl;
			}
		}
		if (num_bad_roots > 0) {
			std::clog << "No. of bad roots = " << num_bad_roots << std::endl;
			throw std::runtime_error(
			    "AR/MA process is not stationary/invertible: some roots lie "
			    "inside unit circle or on its borderline.");
		}
	}

	template <class T>
	T
	ACF_variance(const ACF<T>& acf) {
		return acf(0, 0, 0);
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
		if (!blitz::all(blitz::isfinite(eps))) {
			throw std::runtime_error(
			    "white noise generator produced some NaN/Inf");
		}
		return eps;
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
