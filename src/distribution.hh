#ifndef DISTRIBUTION_HH
#define DISTRIBUTION_HH

#include <algorithm>

#include <blitz/array.h>
#include <gsl/gsl_cdf.h>

#include "statistics.hh"

namespace stats {

	template <class T>
	struct Gaussian {

		Gaussian(T m, T sigma) : _mean(m), _sigma(sigma) {}

		T
		quantile(T f) {
			return gsl_cdf_gaussian_Pinv(f, _sigma) - _mean;
		}

	private:
		T _mean;
		T _sigma;
	};

	template <class T>
	struct Rayleigh {

		Rayleigh(T sigma) : _sigma(sigma) {}

		T
		quantile(T f) {
			return gsl_cdf_rayleigh_Pinv(f, _sigma);
		}

	private:
		T _sigma;
	};

	template <class T>
	struct Weibull {

		Weibull(T a, T b) : _a(a), _b(b) {}

		T
		quantile(T f) {
			return gsl_cdf_weibull_Pinv(f, _a, _b);
		}

	private:
		T _a;
		T _b;
	};

	template <class T, int N, class D>
	T
	distance(D dist, blitz::Array<T, N> rhs, size_t nquantiles = 100) {
		blitz::Array<T, N> data(rhs.shape());
		data = rhs;
		/// 1. Calculate expected quantile values from supplied quantile
		/// function.
		blitz::Array<T, 1> expected(nquantiles);
		for (size_t i = 0; i < nquantiles; ++i) {
			const T f = T(1.0) / T(nquantiles - 1);
			expected(i) = dist.quantile(f);
		}
		/// 2. Calculate real quantiles from data.
		std::sort(data.data(), data.data() + data.numElements());
		blitz::Array<T, 1> real(nquantiles);
		for (size_t i = 0; i < nquantiles; ++i) {
			const T f = T(1.0) / T(nquantiles - 1);
			real(i) = ::stats::quantile(data, f);
		}
		/// 3. Calculate distance between two quantile vectors.
		return std::sqrt(blitz::sum(blitz::pow2(expected - real)));
	}
}

#endif // DISTRIBUTION_HH
