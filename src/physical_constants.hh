#ifndef PHYSICAL_CONSTANTS_HH
#define PHYSICAL_CONSTANTS_HH

#include <cmath>

namespace arma {

	namespace constants {

		template <class T>
		constexpr const T pi = M_PI;

		template <class T>
		constexpr const T _2pi = T(2)*pi<T>;

	}

}

#endif // PHYSICAL_CONSTANTS_HH
