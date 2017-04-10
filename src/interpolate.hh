#ifndef INTERPOLATE_HH
#define INTERPOLATE_HH

#include "types.hh"

namespace arma {

	template <class T>
	Vec3D<T>
	interpolation_polynomial(
		const Vec2D<int>& x1,
		const Vec2D<int>& x2,
		const Vec2D<int>& x3,
		const Array2D<T>& f
	);

	template <class T>
	inline T
	interpolate(const Vec3D<T>& coef, const Vec2D<int>& x) {
		return coef(0)*x(0) + coef(1)*x(1) + coef(2);
	}

	template <class T>
	inline T
	interpolate(
		const Vec2D<int>& x1,
		const Vec2D<int>& x2,
		const Vec2D<int>& x3,
		const Array2D<T>& f,
		const Vec2D<int>& x
	) {
		return interpolate(interpolation_polynomial(x1, x2, x3, f), x);
	}

}

#endif // INTERPOLATE_HH
