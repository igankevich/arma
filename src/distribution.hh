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
			return gsl_cdf_gaussian_Pinv(f, _sigma) + _mean;
		}

		template <int N>
		T
		distance(blitz::Array<T, N> rhs, size_t nquantiles = 100) {
			blitz::Array<T, N> data(rhs.shape());
			data = rhs;
			/// 1. Calculate expected quantile values from supplied quantile
			/// function.
			blitz::Array<T, 1> expected(nquantiles);
			for (size_t i = 0; i < nquantiles; ++i) {
				const T f = T(1.0) / T(nquantiles - 1);
				expected(i) = this->quantile(f);
			}
			/// 2. Calculate real quantiles from data.
			std::sort(data.data(), data.data() + data.numElements());
			blitz::Array<T, 1> real(nquantiles);
			for (size_t i = 0; i < nquantiles; ++i) {
				const T f = T(1.0) / T(nquantiles - 1);
				real(i) = ::stats::quantile(data, f);
			}
			/// 3. Calculate distance between two quantile vectors and compare
			/// it to epsilon.
			return std::sqrt(blitz::sum(blitz::pow2(expected - real)));
			//			if (distance > eps) {
			//				std::clog << "Distributions do not match: distance="
			//<<
			// distance
			//				          << ", eps=" << eps << std::endl;
			//				throw std::runtime_error("invalid distribution");
			//			}
		}

	private:
		T _mean;
		T _sigma;
	};
}

#endif // DISTRIBUTION_HH
