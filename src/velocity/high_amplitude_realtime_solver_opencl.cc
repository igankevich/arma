const char* HARS_SRC = CL_PROGRAM_STRING_DEBUG_INFO ARMA_STRINGIFY(

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

);

