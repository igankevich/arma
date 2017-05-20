#ifndef APMATH_HERMITE_HH
#define APMATH_HERMITE_HH

#include "polynomial.hh"

namespace arma {

	namespace apmath {

		/**
		\brief Constructs symbolic Hermite polynomial of order \f$n\f$.
		\date 2017-05-20
		\author Ivan Gankevich
		*/
		template <class T>
		Polynomial<T>
		hermite_polynomial(int n);

	}

}

#endif // APMATH_HERMITE_HH
