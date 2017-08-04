#ifndef ARMA_HH
#define ARMA_HH

#include <algorithm>  // for any_of, generate
#include <cassert>    // for assert
#include <chrono>     // for duration, steady_clock, steady_clock::...
#include <cmath>      // for isinf, isnan
#include <functional> // for bind
#include <random>     // for mt19937, normal_distribution
#include <stdexcept>  // for runtime_error
#include <ostream>
#include <fstream>
#include <type_traits>

#include <gsl/gsl_complex.h> // for gsl_complex_packed_ptr
#include <gsl/gsl_errno.h>   // for gsl_strerror, ::GSL_SUCCESS
#include <gsl/gsl_poly.h>    // for gsl_poly_complex_solve, gsl_poly_com...

#include "types.hh" // for Array3D, ACF, Shape3D
#include "stats/statistics.hh"
#include "stats/distribution.hh"
#include "stats/qq_graph.hh"
#include "fourier.hh"
#include "blitz.hh"

/// @file
/// File with auxiliary subroutines.


/// Domain-specific classes and functions.
namespace arma {

	/// Check AR (MA) process stationarity (invertibility).
	template <class T, int N>
	void
	validate_process(blitz::Array<T, N> _phi);

	template <class T>
	T
	ACF_variance(const Array3D<T>& acf) {
		return acf(0, 0, 0);
	}

	template <class T>
	T
	approx_wave_height(T variance) {
		return std::sqrt(T(2) * M_PI * variance);
	}

	template <class T>
	T
	approx_wave_period(T variance) {
		return T(4.8) * std::sqrt(approx_wave_height(variance));
	}

}

#endif // ARMA_HH
