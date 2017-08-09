#ifndef STATS_GRAM_CHARLIER_HH
#define STATS_GRAM_CHARLIER_HH

#include <istream>
#include <ostream>
#include <cmath>
#include "physical_constants.hh"

namespace arma {

	namespace stats {

		/**
		\brief Skewed normal distribution.
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

			inline T
			skewness() const noexcept {
				return this->_skewness;
			}

			inline T
			kurtosis() const noexcept {
				return this->_kurtosis;
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

	}

}

#endif // STATS_GRAM_CHARLIER_HH
