#ifndef VOODOO_HH
#define VOODOO_HH

#include <assert.h>      // for assert
#include <blitz/array.h> // for Range, toEnd, shape
#include <cstdlib>       // for abs
#include "types.hh"      // for Array2D, Shape3D

/**
\file
\author Ivan Gankevich
\date 2016-07-26

\brief
Non-optimised parts of the implementation.

\details
There exist more efficient ways to compute autoregressive model
coefficients that take into account autocovariance matrix structure, but they
seem complex and not wide-spread. So, we settled on "keep it simple" approach.
*/

namespace arma {

	namespace generator {

		/**
		\brief Autocovariate matrix generator which uses least squares
		approximations to reduce size of autocovariate function grid.
		*/
		template <class T>
		class AC_matrix_generator_LS {

		private:
			const Array3D<T>& _acf;
			const Shape3D& _arorder;

		public:

			AC_matrix_generator_LS(const Array3D<T>& acf, const Shape3D& ar_order)
			    : _acf(acf), _arorder(ar_order) {}

			Array2D<T>
			AC_matrix_block(int i0, int j0);

			Array2D<T>
			AC_matrix_block(int i0);

			Array2D<T> operator()();

		};

		/**
		\brief Autocovariate matrix generator that reduces the size of ACF to match
		AR model order.
		*/
		template <class T>
		class AC_matrix_generator {

		private:
			const Array3D<T>& _acf;
			const Shape3D& _arorder;

		public:

			AC_matrix_generator(const Array3D<T>& acf, const Shape3D& ar_order)
			    : _acf(acf), _arorder(ar_order) {}

			Array2D<T>
			AC_matrix_block(int i0, int j0);

			Array2D<T>
			AC_matrix_block(int i0);

			Array2D<T> operator()();

		};

		/**
		\brief Matrix generator for moving average model.
		*/
		template <class T>
		class Tau_matrix_generator {

		private:
			Array3D<T> _tau;

		public:

			inline
			Tau_matrix_generator(Array3D<T> tau): _tau(tau) {}

			Array2D<T>
			tau_matrix_block(int i0, int j0);

			Array2D<T>
			tau_matrix_block(int i0);

			Array2D<T> operator()();

		};

	}

}

#endif // VOODOO_HH
