#ifndef STATISTICS_HH
#define STATISTICS_HH

#include <blitz/array.h> // for Array

#include <gsl/gsl_statistics_float.h>
#include <gsl/gsl_statistics_double.h>

namespace arma {

	/// Convenience mappings of C++ templates to GSL functions.
	namespace stats {

		inline float
		mean(const float* data, const size_t n) {
			return gsl_stats_float_mean(data, 1, n);
		}

		inline double
		mean(const double* data, const size_t n) {
			return gsl_stats_mean(data, 1, n);
		}

		template <class T, int N>
		T
		mean(blitz::Array<T, N> rhs) {
			return mean(rhs.data(), rhs.numElements());
		}

		inline float
		variance(const float* data, const size_t n) {
			return gsl_stats_float_variance(data, 1, n);
		}

		inline double
		variance(const double* data, const size_t n) {
			return gsl_stats_variance(data, 1, n);
		}

		template <class T, int N>
		T
		variance(blitz::Array<T, N> rhs) {
			return variance(rhs.data(), rhs.numElements());
		}

		inline float
		skew(const float* data, const size_t n) {
			return gsl_stats_float_skew(data, 1, n);
		}

		inline double
		skew(const double* data, const size_t n) {
			return gsl_stats_skew(data, 1, n);
		}

		template <class T, int N>
		T
		skew(blitz::Array<T, N> rhs) {
			return skew(rhs.data(), rhs.numElements());
		}

		inline float
		kurtosis(const float* data, const size_t n) {
			return gsl_stats_float_kurtosis(data, 1, n);
		}

		inline double
		kurtosis(const double* data, const size_t n) {
			return gsl_stats_kurtosis(data, 1, n);
		}

		template <class T, int N>
		T
		kurtosis(blitz::Array<T, N> rhs) {
			return kurtosis(rhs.data(), rhs.numElements());
		}

		inline float
		quantile(const float* data, const size_t n, const float f) {
			return gsl_stats_float_quantile_from_sorted_data(data, 1, n, f);
		}

		inline double
		quantile(const double* data, const size_t n, const double f) {
			return gsl_stats_quantile_from_sorted_data(data, 1, n, f);
		}

		template <class T, int N>
		T
		quantile(blitz::Array<T, N> rhs, T f) {
			return quantile(rhs.data(), rhs.numElements(), f);
		}

		inline float
		stdev(const float* data, const size_t n) {
			return gsl_stats_float_sd(data, 1, n);
		}

		inline double
		stdev(const double* data, const size_t n) {
			return gsl_stats_sd(data, 1, n);
		}

		template <class T, int N>
		T
		stdev(blitz::Array<T, N> rhs) {
			return stdev(rhs.data(), rhs.numElements());
		}

	}

}

#endif // STATISTICS_HH
