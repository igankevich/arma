#include "series.hh"

#include "apmath/factorial.hh"
#include "apmath/hermite.hh"
#include "apmath/polynomial.hh"

#include <limits>
#include <cmath>
#ifndef NDEBUG
#include <iostream>
#endif

template <class T>
blitz::Array<T, 1>
arma::nonlinear::gram_charlier_expand(
	blitz::Array<T, 1> a,
	const int order,
	const T acf_variance
) {
	typedef ::arma::apmath::Polynomial<T> poly_type;
	using ::arma::apmath::hermite_polynomial;
	using ::arma::apmath::factorial;
	using std::abs;

	blitz::Array<T,1> c(order);
	T sum_c = 0;
	T f = 1;
	T err = std::numeric_limits<T>::max();
	int trim = 0;
	for (int m=0; m<order; ++m) {
		/**
		1. Calculate series coefficients \f$C\f$:
		\f[
			C_i = p_0 +
			\begin{cases}
			\sum\limits_{n=2}^{m} p_n (n-1)!!
				& \text{if }n\text{ is even},\\
			0   & \text{if }n\text{ is odd},
			\end{cases}
		\f]
		where \f$m\f$ is the current polynomial order and
		\f$p_n\f$ is a coefficient before \f$n\f$-th order
		polynomial term.
		*/
		poly_type y = poly_type(a) * hermite_polynomial<T>(m);
		T sum2 = y(0);
		const int yorder = y.order();
		for (int i=2; i<=yorder; i+=2) {
			sum2 += y(i)*factorial<T>(i-1, 2);
		}
		c[m] = sum2;

		/**
		2. Calculate error:
		\f[
			e_m = \left|
				\sum\limits_{n=0}^{m} \frac{C_i^2}{(n+1)!}
				- \gamma_{\vec{0}}
			\right|.
		\f]
		*/
		sum_c += c(m)*c(m)/f;
		f *= (m+1);
		const T e = abs(acf_variance - sum_c);
		// criteria could be: abs(T(1) - sum_c)

		// determine minimum error
		if (e < err) {
			err = e;
			trim = m+1;
		}
		#ifndef NDEBUG
		std::clog << "err = " << e << std::endl;
		#endif
	}
	std::clog << "trim = " << trim << std::endl;
	c.resizeAndPreserve(trim);
	return c;
}

template blitz::Array<ARMA_REAL_TYPE, 1>
arma::nonlinear::gram_charlier_expand(
	blitz::Array<ARMA_REAL_TYPE, 1> a,
	const int order,
	const ARMA_REAL_TYPE acf_variance
);
