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
	const int max_order,
	const T acf_variance,
	T& out_err
) {
	typedef ::arma::apmath::Polynomial<T> poly_type;
	using ::arma::apmath::hermite_polynomial;
	using ::arma::apmath::factorial;
	using std::abs;
	using std::isfinite;
	using blitz::Range;

	blitz::Array<T,1> c(max_order);
	T err;
	T sum_c = 0;
	T f = 1;
	T e = std::numeric_limits<T>::max();
	int m = 0;
	do {
		err = e;
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
		c(m) = sum2;

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
		e = abs(acf_variance - sum_c);
		// criteria could be: abs(T(1) - sum_c)

		//#ifndef NDEBUG
		//std::clog << "m=" << m << ",e=" << e << std::endl;
		//#endif
	} while (e < err && ++m < max_order);
	c.resizeAndPreserve(m);
	out_err = err;
	return c;
}

template blitz::Array<ARMA_REAL_TYPE, 1>
arma::nonlinear::gram_charlier_expand(
	blitz::Array<ARMA_REAL_TYPE, 1> a,
	const int max_order,
	const ARMA_REAL_TYPE acf_variance,
	ARMA_REAL_TYPE& err
);
