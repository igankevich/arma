#ifndef NONLINEAR_SERIES_HH
#define NONLINEAR_SERIES_HH

#include <blitz/array.h>

namespace arma {

	namespace nonlinear {

		/**
		\brief Expands ACF into Gram---Charlier series.
		\date 2017-05-20
		\author Ivan Gankevich
		\param[in] a polynomial coefficients of a transformed cumulative
		             distribution function
		\param[in] max_order maximal expansion order
		\param[in] acf_variance ACF variance \f$\gamma_{\vec{0}}\f$
		\param[out] err error of the resulting approximation
		\see transform_CDF
		*/
		template <class T>
		blitz::Array<T, 1>
		gram_charlier_expand(
			blitz::Array<T, 1> a,
			const int max_order,
			const T acf_variance,
			T& err
		);

	}

}

#endif // NONLINEAR_SERIES_HH
