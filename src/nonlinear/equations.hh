#ifndef NONLINEAR_EQUATIONS_HH
#define NONLINEAR_EQUATIONS_HH

#include <blitz/array.h>

namespace arma {

	namespace nonlinear {

		/**
		\brief Defines equality between cumulative distribution functions
		       at a specified point.
		\date 2017-05-20
		\author Ivan Gankevich
		*/
		template<class T, class CDF>
		class Equation_CDF {

			/// CDF (LHS of the equation).
			CDF _cdf;

			/// Value of CDF at a specified point (RHS of the equation).
			T _cdfy;

		public:
			inline
			Equation_CDF(CDF func, T y) noexcept:
			_cdf(func),
			_cdfy(y)
			{}

			inline T
			operator()(T x) {
				return _cdf.cdf(x) - _cdfy;
			}

		};

		/**
		\brief Defines equality between ACF and its Gram---Charlier
		       series expansion at a specified point.
		\date 2017-05-20
		\author Ivan Gankevich
		*/
		template<class T>
		class Equation_ACF {

			typedef blitz::Array<T,1> array_type;

			/// Coefficients of the expansion.
			const array_type& _coef;

			/// Value of ACF at a specified point.
			T _acfy;

		public:
			inline
			Equation_ACF(const array_type& coefs, T y):
			_coef(coefs),
			_acfy(y)
			{}

			inline T
			operator()(T x) const {
				T sum = 0;
				T f = 1;
				T x2 = 1;
				const int n = _coef.size();
				for (int i=0; i<n; ++i) {
					const T c_i = _coef(i);
					sum += c_i*c_i*x2/f;
					f *= (i+1);
					x2 *= x;
				}
				return sum - _acfy;
			}
		};

	}

}

#endif // NONLINEAR_EQUATIONS_HH
