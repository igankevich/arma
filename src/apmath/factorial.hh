#ifndef APMATH_FACTORIAL_HH
#define APMATH_FACTORIAL_HH

namespace arma {

	namespace apmath {

		/**
		\brief Computes factorial.
		\date 2017-05-20
		\author Ivan Gankevich

		\param[in] x a number for which to compute factorial
		\param[in] p factorial order

		\f$p=1\f$ --- factorial,
		\f$p=2\f$ --- double factorial,
		\f$p>2\f$ --- multifactorial.
		*/
		template <class T>
		T
		factorial(T x, const T p=1) {
			T m = 1;
			while (x > 1) {
				m *= x;
				x -= p;
			}
			return m;
		}

	}

}

#endif // APMATH_FACTORIAL_HH
