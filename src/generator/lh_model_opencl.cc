const char* LH_MODEL_SRC = CL_PROGRAM_STRING_DEBUG_INFO ARMA_STRINGIFY(
	typedef ARMA_REAL_TYPE T;
	kernel void
	generate_surface(int a, T* b) {
		b[get_global_id(0)] = a;
	}
);
