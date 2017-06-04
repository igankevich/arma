#ifndef STATS_GAUSSIAN_HH
#define STATS_GAUSSIAN_HH

#include <istream>
#include <ostream>
#include <gsl/gsl_cdf.h>

namespace arma {

	namespace stats {

		/// \brief Normal distribution.
		template <class T>
		struct Gaussian {

			Gaussian() = default;
			Gaussian(const Gaussian&) = default;
			Gaussian(Gaussian&&) = default;

			explicit
			Gaussian(T m, T sigma):
			_mean(m),
			_sigma(sigma)
			{}

			inline T
			quantile(T f) {
				return gsl_cdf_gaussian_Pinv(f, _sigma) + _mean;
			}

			inline T
			cdf(T f) {
				return gsl_cdf_gaussian_P(f - _mean, _sigma);
			}

			T mean() const noexcept { return _mean; }
			T stdev() const noexcept { return _sigma; }

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, Gaussian<X>& rhs);

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const Gaussian<X>& rhs);

		private:
			T _mean;
			T _sigma;
		};

		template <class T>
		std::istream&
		operator>>(std::istream& in, Gaussian<T>& rhs);

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const Gaussian<T>& rhs);

	}

}

#endif // STATS_GAUSSIAN_HH
