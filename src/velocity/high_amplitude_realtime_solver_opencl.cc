const char* HARTS_SRC = CL_PROGRAM_STRING_DEBUG_INFO ARMA_STRINGIFY(

	#define NEW_VEC_IMPL(x,y) x##y
	#define NEW_VEC(x,y) NEW_VEC_IMPL(x, y)

	typedef ARMA_REAL_TYPE T;
	typedef NEW_VEC(ARMA_REAL_TYPE,2) T2;
	typedef NEW_VEC(ARMA_REAL_TYPE,3) T3;
	typedef NEW_VEC(ARMA_REAL_TYPE,4) T4;

	typedef union {
		T3 vec;
		T elem[sizeof(T3)];
	} T3_union;

	typedef union {
		int3 vec;
		int elem[sizeof(int3)];
	} int3_union;

	kernel void
	compute_window_function(
		const T3 grid_length,
		const T h,
		global T* result
	) {
		const int nz = get_global_size(0);
		const int nkx = get_global_size(1);
		const int nky = get_global_size(2);
		const int i = get_global_id(0);
		const int j = get_global_id(1);
		const int k = get_global_id(2);
		const T z = grid_length.s0 / (nz-1) * i;
		const T kx = grid_length.s1 / (nkx-1) * j;
		const T ky = grid_length.s2 / (nky-1) * k;
		const T l = 2 * M_PI * length((T2)(kx,ky));
		const T numerator = cosh(l*(z + h));
		const T denominator = l*cosh(l*h);
		result[i*nkx*nky + j*nky + k] = 4 * M_PI * numerator / denominator;
	}

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
	}

);

