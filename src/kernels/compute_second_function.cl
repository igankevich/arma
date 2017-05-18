#include "common.h"

kernel void
compute_second_function(
	global const T* zeta,
	const T3_union delta,
	const int dimension,
	global T* result
) {
	#define IDX(un) (un.elem[0]*nx*ny + un.elem[1]*ny + un.elem[2])
	const int nt = get_global_size(0);
	const int nx = get_global_size(1);
	const int ny = get_global_size(2);
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int k = get_global_id(2);
	const int3_union idx0 = { .vec = (int3)(i, j, k) };
	const int off0 = IDX(idx0);
	const T zeta0 = zeta[off0];
	const T denominator = 2*delta.elem[dimension];
	if (get_global_id(dimension) == 0) {
		int3_union idx1 = { .vec = idx0.vec };
		++idx1.elem[dimension];
		int3_union idx2 = { .vec = idx1.vec };
		++idx2.elem[dimension];
		const T zeta1 = zeta[IDX(idx1)];
		const T zeta2 = zeta[IDX(idx2)];
		result[off0] = (-zeta2 + 4*zeta1 - 3*zeta0) / denominator;
	} else if (get_global_id(dimension) == get_global_size(dimension) - 1) {
		int3_union idx1 = { .vec = idx0.vec };
		--idx1.elem[dimension];
		int3_union idx2 = { .vec = idx1.vec };
		--idx2.elem[dimension];
		const T zeta1 = zeta[IDX(idx1)];
		const T zeta2 = zeta[IDX(idx2)];
		result[off0] = (zeta0 - 4*zeta1 + zeta2) / denominator;
	} else {
		int3_union idx1 = { .vec = idx0.vec };
		--idx1.elem[dimension];
		int3_union idx2 = { .vec = idx0.vec };
		++idx2.elem[dimension];
		const T zeta1 = zeta[IDX(idx1)];
		const T zeta2 = zeta[IDX(idx2)];
		result[off0] = (zeta2 - zeta1) / denominator;
	}
	#undef IDX
}

