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

	template <class T, int N>
	T
	abs_max(blitz::Array<T, N> rhs) {
		return std::max(std::abs(blitz::max(rhs)), std::abs(blitz::min(rhs)));
	}

	template <class T>
	struct Quantile_graph {

		template <int N, class D>
		Quantile_graph(D dist, blitz::Array<T, N> rhs, size_t nquantiles = 100)
		    : _expected(nquantiles), _real(nquantiles) {
			blitz::Array<T, N> data = rhs.copy();
			/// 1. Calculate expected quantile values from supplied quantile
			/// function.
			for (size_t i = 0; i < nquantiles; ++i) {
				const T f = T(1.0) / T(nquantiles - 1) * T(i);
				_expected(i) = dist.quantile(f);
			}
			/// 2. Calculate real quantiles from data.
			std::sort(data.data(), data.data() + data.numElements());
			for (size_t i = 0; i < nquantiles; ++i) {
				const T f = T(1.0) / T(nquantiles - 1) * T(i);
				_real(i) = ::stats::quantile(data, f);
			}
		}

		/// Calculate distance between two quantile vectors.
		T
		distance() const {
			using blitz::sum;
			using blitz::abs;
			/// 1. Omit first and last quantiles if they are not finite in
			/// expected distribution.
			int x0 = !std::isfinite(_expected(0)) ? 1 : 0;
			int x1 = !std::isfinite(_expected(_expected.size() - 1))
			             ? _expected.size() - 2
			             : _expected.size() - 1;
			blitz::Range r(x0, x1);
			/// 2. Rescale all values to \f$[0,1]\f$ range.
			const T scale = std::max(abs_max(_expected(r)), abs_max(_real(r)));
			const int nquantiles = x1 - x0 + 1;
			return sum(abs(_expected(r) - _real(r))) / scale / T(nquantiles);
		}

		friend std::ostream& operator<<(std::ostream& out,
		                                const Quantile_graph& rhs) {
			std::transform(
			    rhs._expected.begin(), rhs._expected.end(), rhs._real.begin(),
			    std::ostream_iterator<QPair<T>>(out, "\n"),
			    [](float lhs, float rhs) { return QPair<T>(lhs, rhs); });
			return out;
		}

	private:
		template <class X>
		struct QPair {

			QPair(X a, X b) : first(a), second(b) {}

			friend std::ostream& operator<<(std::ostream& out,
			                                const QPair& rhs) {
				return out << std::setw(20) << rhs.first << std::setw(20)
				           << rhs.second;
			}

			X first;
			X second;
		};
		blitz::Array<T, 1> _expected;
		blitz::Array<T, 1> _real;
	};
}

#endif // DISTRIBUTION_HH
