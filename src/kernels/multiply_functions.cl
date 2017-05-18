#include "common.h"

kernel void
multiply_functions(
	global T2* sfunc,
	global const T* wfunc
) {
	const int nz = get_global_size(0);
	const int nkx = get_global_size(1);
	const int nky = get_global_size(2);
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int k = get_global_id(2);
	const int off = i*nkx*nky + j*nky + k;
	sfunc[off] *= wfunc[off];
}

