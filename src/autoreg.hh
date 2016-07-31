#ifndef AUTOREG_HH
#define AUTOREG_HH

#include <blitz/array.h> // for Array
#include <algorithm>     // for any_of, generate
#include <cassert>       // for assert
#include <chrono>        // for duration, steady_clock, steady_clock::...
#include <cmath>         // for isinf, isnan
#include <functional>    // for bind
#include <random>        // for mt19937, normal_distribution
#include <stdexcept>     // for runtime_error
#include "types.hh"      // for Zeta, ACF, size3

/// @file
/// File with auxiliary subroutines.

/// Domain-specific classes and functions.
namespace autoreg {

	template <class T>
	T
	ACF_variance(const ACF<T>& acf) {
		return acf(0, 0, 0);
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
