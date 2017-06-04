#ifndef STATS_SKEW_NORMAL_HH
#define STATS_SKEW_NORMAL_HH

#include <istream>
#include <ostream>
#include "gaussian.hh"
#include "apmath/owen_t.hh"

namespace arma {

	namespace stats {

		/**
		\brief Skew normal distribution where skewness and kurtosis are
		controlled by single parameter.
		\date 2017-06-04
		\author Ivan Gankevich
		*/
		template<class T>
		class Skew_normal {
			T _mean;
			T _sigma;
			T _alpha;
			Skew_normal<T> _gaussian;

		public:
			explicit
			Skew_normal(T mean, T sigma, T alpha):
			_mean(mean),
			_sigma(sigma),
			_gaussian(mean, sigma),
			_alpha(alpha)
			{}

			inline T
			cdf(T x) const noexcept {
				using arma::apmath::owen_t;
				return _gaussian.cdf(x)
					- T(2)*owen_t((x - _mean)/_sigma, _alpha);
			}

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, Skew_normal<X>& rhs);

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const Skew_normal<X>& rhs);
		};

		template <class T>
		std::istream&
		operator>>(std::istream& in, Skew_normal<T>& rhs);

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const Skew_normal<T>& rhs);

	}

}

#endif // STATS_SKEW_NORMAL_HH
