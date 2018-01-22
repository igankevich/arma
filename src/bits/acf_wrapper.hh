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

		private:
			typedef blitz::TinyVector<T,3> alpha_type;
			typedef blitz::TinyVector<T,2> beta_type;

		private:
			Discrete_function<T,3>& _acf;
			T _amplitude = T(1);
			T _velocity = T(1);
			/// Decay factor (t,x,y).
			alpha_type _alpha = alpha_type(0.06, 0.06, 0.06);
			/// Wave numbers (x,y).
			beta_type _beta = beta_type(0.8, 0.0);

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
