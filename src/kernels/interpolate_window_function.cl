#include "common.h"
#include "harts.h"
#include "interpolation_polynomial.cl"

kernel void
interpolate_window_function(global T* result) {
	const int nz = get_global_size(0);
	const int nkx = get_global_size(1);
	const int nky = get_global_size(2);
	const int i = get_global_id(0);
	const int j = rotate_right(get_global_id(1), nkx);
	const int k = rotate_right(get_global_id(2), nky);
	const int idx = i*nkx*nky + j*nky + k;
	const int j0 = rotate_right(0, nkx);
	const int j1 = j0 + 1;
	const int j2 = j0 + 2;
	const int k0 = rotate_right(0, nky);
	const int k1 = k0 + 1;
	const int k2 = k0 + 2;
	if (j == j0 && k == k0) {
		result[idx] = result[i*nkx*nky + (j+1)*nky + (k+1)];
	}
	if (j != j0 && k == k0) {
		result[idx] = interpolate(
			(int2)((j-1+nkx)%nkx,k1),
			(int2)(j,k1),
			(int2)((j-1+nkx)%nkx,k2),
			result,
			i, nz, nkx, nky,
			(int2)(j,k0)
		);
	}
	if (j == j0 && k != k0) {
		result[idx] = interpolate(
			(int2)(j1,(k-1+nky)%nky),
			(int2)(j1,k),
			(int2)(j2,(k-1+nky)%nky),
			result,
			i, nz, nkx, nky,
			(int2)(j0,k)
		);
	}
	//if (!isfinite(result[idx])) {
	//	printf("result inf at %i,%i,%i\n", i, j, k);
	//}
}
