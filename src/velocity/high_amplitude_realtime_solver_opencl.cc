const char* HARTS_SRC = CL_PROGRAM_STRING_DEBUG_INFO ARMA_STRINGIFY(

	#define NEW_VEC_IMPL(x,y) x##y
	#define NEW_VEC(x,y) NEW_VEC_IMPL(x, y)

	typedef ARMA_REAL_TYPE T;
	typedef NEW_VEC(ARMA_REAL_TYPE,2) T2;
	typedef NEW_VEC(ARMA_REAL_TYPE,4) T3; // use cl_double4 instead of cl_double3
	typedef NEW_VEC(ARMA_REAL_TYPE,4) T4;

	typedef union {
		T3 vec;
		T elem[sizeof(T3)];
	} T3_union;

	typedef union {
		int3 vec;
		int elem[sizeof(int3)];
	} int3_union;

	int
	rotate_right(const int idx, const int n) {
		return (idx + n/2)%n;
	}

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

);

