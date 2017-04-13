const char* HARTS_SRC = CL_PROGRAM_STRING_DEBUG_INFO ARMA_STRINGIFY(

	#define NEW_VEC_IMPL(x,y) x##y
	#define NEW_VEC(x,y) NEW_VEC_IMPL(x, y)

	typedef ARMA_REAL_TYPE T;
	typedef NEW_VEC(ARMA_REAL_TYPE,2) T2;
	typedef NEW_VEC(ARMA_REAL_TYPE,3) T3;
	typedef NEW_VEC(ARMA_REAL_TYPE,4) T4;

	kernel void
	compute_velocity_field(
		global const T* phi,
		const T4 outgrid_length
	) {
	}

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

);

