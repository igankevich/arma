#ifndef DISTRIBUTION_HH
#define DISTRIBUTION_HH

#include <istream>
#include <ostream>
#include <gsl/gsl_cdf.h>
#include "physical_constants.hh"

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

		/// \brief Weibull distribution.
		template <class T>
		struct Weibull {

			Weibull(T a, T b):
			_a(a),
			_b(b)
			{}

			T
			quantile(T f) {
				return gsl_cdf_weibull_Pinv(f, _a, _b);
			}

		private:
			T _a; //< lambda
			T _b; //< k
		};

		/**
		\brief Skew normal distribution.
		\date 2017-05-20
		\author Ivan Gankevich

		\details Gram---Charlier series approximation with configurable
		skewness and kurtosis.
		*/
		template<class T>
		class Gram_Charlier {
			T _skewness;
			T _kurtosis;

		public:
			Gram_Charlier() = default;
			Gram_Charlier(const Gram_Charlier&) = default;
			Gram_Charlier(Gram_Charlier&&) = default;

			inline explicit
			Gram_Charlier(T skew, T kurt) noexcept:
			_skewness(skew),
			_kurtosis(kurt)
			{}

			inline T
			cdf(T x) const noexcept {
				using constants::_2pi;
				using constants::sqrt2pi;
				using constants::sqrt2;
				return std::exp(T(-0.5)*x*x)*(_kurtosis*(T(3)*x - x*x*x)
					+ _skewness*(T(4) - T(4)*x*x) + T(3)*x*x*x - T(9)*x)
					/ (T(24)*sqrt2pi<T>) + T(0.5)*std::erf(x/sqrt2<T>) + T(0.5);
			}

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, Gram_Charlier<X>& rhs);

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const Gram_Charlier<X>& rhs);

		};

		template <class T>
		std::istream&
		operator>>(std::istream& in, Gram_Charlier<T>& rhs);

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const Gram_Charlier<T>& rhs);

		enum struct Characteristic {
			Wave_height,
			Wave_length,
			Crest_length,
			Wave_period,
			Wave_slope,
			Threedimensionality
		};

		/// \brief Determines Weibull distribution shape parameter
		/// for different wave characteristics.
		template <Characteristic c, class T>
		struct Weibull_shape {};

		/// \copydoc Weibull_shape
		template <class T>
		struct Weibull_shape<Characteristic::Wave_height, T> {
			constexpr static const T shape = 2.0;
		};

		/// \copydoc Weibull_shape
		template <class T>
		struct Weibull_shape<Characteristic::Wave_length, T> {
			constexpr static const T shape = 2.3;
		};

		/// \copydoc Weibull_shape
		template <class T>
		struct Weibull_shape<Characteristic::Crest_length, T> {
			constexpr static const T shape = 2.3;
		};

		/// \copydoc Weibull_shape
		template <class T>
		struct Weibull_shape<Characteristic::Wave_period, T> {
			constexpr static const T shape = 3.0;
		};

		/// \copydoc Weibull_shape
		template <class T>
		struct Weibull_shape<Characteristic::Wave_slope, T> {
			constexpr static const T shape = 2.5;
		};

		/// \copydoc Weibull_shape
		template <class T>
		struct Weibull_shape<Characteristic::Threedimensionality, T> {
			constexpr static const T shape = 2.5;
		};

		/// \brief Distribution function of wave height, period, length etc.
		template <class T, Characteristic C>
		struct Wave_characteristic_distribution : public Weibull<T> {
			Wave_characteristic_distribution(T mean)
			    : Weibull<T>(mean / std::tgamma(T(1) + T(1) / shape), shape) {}
			constexpr static const T shape = Weibull_shape<C, T>::shape;
		};

		template <class T>
		using Wave_heights_dist =
		    Wave_characteristic_distribution<T, Characteristic::Wave_height>;

		template <class T>
		using Wave_lengths_dist =
		    Wave_characteristic_distribution<T, Characteristic::Wave_length>;

		template <class T>
		using Crest_lengths_dist =
		    Wave_characteristic_distribution<T, Characteristic::Crest_length>;

		template <class T>
		using Wave_periods_dist =
		    Wave_characteristic_distribution<T, Characteristic::Wave_period>;

		template <class T>
		using Wave_slopes_dist =
		    Wave_characteristic_distribution<T, Characteristic::Wave_slope>;

		template <class T>
		using Threedimensionality_dist =
		    Wave_characteristic_distribution<T, Characteristic::Threedimensionality>;

	}

}

#endif // DISTRIBUTION_HH
