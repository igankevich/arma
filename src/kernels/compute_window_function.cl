#include "common.h"
#include "harts.h"

kernel void
compute_window_function(
	const T3 grid_length,
	const T min_z,
	const T h,
	global T* result
) {
	const int nz = get_global_size(0);
	const int nkx = get_global_size(1);
	const int nky = get_global_size(2);
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int k = get_global_id(2);
	const T z = min_z + grid_length.s0 / (nz-1) * i;
	const T kx = grid_length.s1 / (nkx-1) * rotate_right(j, nkx);
	const T ky = grid_length.s2 / (nky-1) * rotate_right(k, nky);
	const T l = 2 * M_PI * length((T2)(kx,ky));
	const T numerator = cosh(l*(z + h));
	const T denominator = l*cosh(l*h);
	result[i*nkx*nky + j*nky + k] = 4 * M_PI * numerator / denominator;
}


