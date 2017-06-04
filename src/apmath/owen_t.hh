#ifndef APMATH_OWEN_T_HH
#define APMATH_OWEN_T_HH

namespace arma {

	namespace apmath {


		/**
		\brief Computes Owen's \f$T\f$-function.
		\date 2017-06-04
		\author Ivan Gankevich

		Uses Gauss integration formula from \cite patefield2000fast
		\f[
			T(h, \alpha) =
			\frac{\alpha}{2\pi}
			\sum\limits_{i=1}^{m}
			\omega_i \exp\left[
				-\frac{1}{2}h^2 \left( 1 + \alpha^2x_i^2 \right)
			\right]
			\frac{1}{1 + \alpha^2x_i^2}
		\f]
		with weights \f$\omega_i\f$ computed via Legendre polynomials
		\cite abramowitz1972handbook (eq. 25.4.40):
		\f[
			\omega_i = \frac{2}{\left(1 - x_i^2 \right) P_n'(x_i)^2},
		\f]
		where \f$P_n\f$ is Legendre polynomial of order \f$n\f$. Abscissas
		\f$x_i\f$ are the roots of this polynomial. Uses 10<sup>th</sup>
		order polynomial.
		*/
		template <class T>
		T
		owen_t(T h, T alpha);

	}

}

#endif // APMATH_OWEN_T_HH
