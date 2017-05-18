#include "common.h"

kernel void
create_vector_field(
	const T3 grid_length,
	const T min_z,
	global T* vphi,
	global const T* phi
) {
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int k = get_global_id(2);
	const int nz = get_global_size(0);
	const int nkx = get_global_size(1);
	const int nky = get_global_size(2);
	const int off = i*nkx*nky + j*nky + k;
	const T z = min_z + grid_length.s0 / (nz-1) * i;
	const T x = grid_length.s1 / (nkx-1) * j;
	const T y = grid_length.s2 / (nky-1) * k;
	vphi[3*off + 0] = x;
	vphi[3*off + 1] = y;
	vphi[3*off + 2] = phi[off];
}
