#include "common.h"

T3
interpolation_polynomial(
	const int2 x1,
	const int2 x2,
	const int2 x3,
	global const T* f,
	const int i,
	const int nz,
	const int nkx,
	const int nky
) {
	const int idx1 = i*nkx*nky + x1.s0*nky + x1.s1;
	const int idx2 = i*nkx*nky + x2.s0*nky + x2.s1;
	const int idx3 = i*nkx*nky + x3.s0*nky + x3.s1;
	return (T3)(
 (f[idx2]*x1.s1 - f[idx3]*x1.s1 - f[idx1]*x2.s1 +
	f[idx3]*x2.s1 + f[idx1]*x3.s1 - f[idx2]*x3.s1)/
  (x2.s0*x1.s1 - x3.s0*x1.s1 - x1.s0*x2.s1 +
	x3.s0*x2.s1 + x1.s0*x3.s1 - x2.s0*x3.s1),

(f[idx3]*(-x1.s0 + x2.s0) +
	f[idx2]*(x1.s0 - x3.s0) +
	f[idx1]*(-x2.s0 + x3.s0))/
  (x3.s0*(x1.s1 - x2.s1) +
	x1.s0*(x2.s1 - x3.s1) +
	x2.s0*(-x1.s1 + x3.s1)),

(f[idx3]*x2.s0*x1.s1 -
	f[idx2]*x3.s0*x1.s1 - f[idx3]*x1.s0*x2.s1 +
	f[idx1]*x3.s0*x2.s1 + f[idx2]*x1.s0*x3.s1 -
	f[idx1]*x2.s0*x3.s1)/
  (x2.s0*x1.s1 - x3.s0*x1.s1 - x1.s0*x2.s1 +
	x3.s0*x2.s1 + x1.s0*x3.s1 - x2.s0*x3.s1),
  		0
	);
}

T
interpolate(
	const int2 x1,
	const int2 x2,
	const int2 x3,
	global const T* f,
	const int i,
	const int nz,
	const int nkx,
	const int nky,
	const int2 x
) {
	T3 coef = interpolation_polynomial(x1, x2, x3, f, i, nz, nkx, nky);
	return coef.s0*x.s0 + coef.s1*x.s1 + coef.s2;
}
