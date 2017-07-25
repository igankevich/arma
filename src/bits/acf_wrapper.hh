#ifndef BITS_ACF_WRAPPER_HH
#define BITS_ACF_WRAPPER_HH

#include <istream>
#include <string>
#include "types.hh"
#include "discrete_function.hh"

namespace arma {

	namespace bits {

		/// Helper class to init ACF by name.
		template<class T>
		class ACF_wrapper {
			Discrete_function<T,3>& _acf;

		public:
			inline explicit
			ACF_wrapper(Discrete_function<T,3>& acf):
			_acf(acf)
			{}

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, ACF_wrapper<X>& rhs);

		};

		template <class T>
		std::istream&
		operator>>(std::istream& in, ACF_wrapper<T>& rhs);

	}

}

#endif // BITS_ACF_WRAPPER_HH
