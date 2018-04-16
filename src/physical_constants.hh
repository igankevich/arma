#ifndef PHYSICAL_CONSTANTS_HH
#define PHYSICAL_CONSTANTS_HH

#include <cmath>

namespace arma {

	/// Physical and mathematical constants.
	namespace constants {

		template <class T>
		constexpr const T pi = M_PI;

		template <class T>
		constexpr const T _2pi = T(2)*pi<T>;

		template <class T>
		constexpr const T pi_div_2 = T(0.5)*pi<T>;

		template <class T>
		constexpr const T sqrt2pi = T(2.506628274631);

		template <class T>
		constexpr const T sqrt2 = T(1.4142135623731);

		template <class T>
		constexpr const T g = T(9.82);

	}

}

#endif // PHYSICAL_CONSTANTS_HH
