#include "interpolate.hh"

template <class T>
arma::Vec3D<T>
arma::interpolation_polynomial(
	const Vec2D<int>& x1,
	const Vec2D<int>& x2,
	const Vec2D<int>& x3,
	const Array2D<T>& f
) {
	return Vec3D<T>(
 (f(x2)*x1(1) - f(x3)*x1(1) - f(x1)*x2(1) +
	f(x3)*x2(1) + f(x1)*x3(1) - f(x2)*x3(1))/
  (x2(0)*x1(1) - x3(0)*x1(1) - x1(0)*x2(1) +
	x3(0)*x2(1) + x1(0)*x3(1) - x2(0)*x3(1)),

(f(x3)*(-x1(0) + x2(0)) +
	f(x2)*(x1(0) - x3(0)) +
	f(x1)*(-x2(0) + x3(0)))/
  (x3(0)*(x1(1) - x2(1)) +
	x1(0)*(x2(1) - x3(1)) +
	x2(0)*(-x1(1) + x3(1))),
(f(x3)*x2(0)*x1(1) -
	f(x2)*x3(0)*x1(1) - f(x3)*x1(0)*x2(1) +
	f(x1)*x3(0)*x2(1) + f(x2)*x1(0)*x3(1) -
	f(x1)*x2(0)*x3(1))/
  (x2(0)*x1(1) - x3(0)*x1(1) - x1(0)*x2(1) +
	x3(0)*x2(1) + x1(0)*x3(1) - x2(0)*x3(1)));
}

template arma::Vec3D<ARMA_REAL_TYPE>
arma::interpolation_polynomial(
	const Vec2D<int>& x1,
	const Vec2D<int>& x2,
	const Vec2D<int>& x3,
	const Array2D<ARMA_REAL_TYPE>& f
);
