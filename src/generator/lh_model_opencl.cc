const char* LH_MODEL_SRC = CL_PROGRAM_STRING_DEBUG_INFO R"OpenCL(

	#define NEW_VEC_IMPL(x,y) x##y
	#define NEW_VEC(x,y) NEW_VEC_IMPL(x, y)

	typedef ARMA_REAL_TYPE T;
	typedef NEW_VEC(ARMA_REAL_TYPE,2) T2;
	typedef NEW_VEC(ARMA_REAL_TYPE,3) T3;

	kernel void
	generate_surface(
		global const T* coef,
		global const T* eps,
		global T* zeta,
		const T2 spec_domain_lbound,
		const T2 spec_domain_ubound,
		const int2 spec_domain_npatches,
		const T3 outgrid_length
	) {
		const int nt = get_global_size(0);
		const int nx = get_global_size(1);
		const int ny = get_global_size(2);
		const int i = get_global_id(0);
		const int j = get_global_id(1);
		const int k = get_global_id(2);
		const T t = outgrid_length.s0 / (nt-1) * i;
		const T x = outgrid_length.s1 / (nx-1) * j;
		const T y = outgrid_length.s2 / (ny-1) * k;
		const int nomega = spec_domain_npatches.s0;
		const int ntheta = spec_domain_npatches.s1;
		const T domega = (spec_domain_ubound.s0 - spec_domain_lbound.s0) / nomega;
		const T dtheta = (spec_domain_ubound.s1 - spec_domain_lbound.s1) / ntheta;
		const T g = 9.8;
		T sum = 0;
		for (int l=0; l<nomega; ++l) {
			for (int m=0; m<ntheta; ++m) {
				const T omega = l*domega;
				const T theta = m*dtheta;
				const T omega_squared = omega*omega;
				const T k_x = omega_squared*cos(theta)/g;
				const T k_y = omega_squared*sin(theta)/g;
				const int idx = l*ntheta + m;
				sum += coef[idx]*cos(k_x*x + k_y*y - omega*t + eps[idx]);
			}
		}
		zeta[i*nx*ny + j*ny + k] = sum;
	}
)OpenCL";
