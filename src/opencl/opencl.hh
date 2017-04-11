#ifndef OPENCL_OPENCL_HH
#define OPENCL_OPENCL_HH

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

namespace arma {

	namespace opencl {

		cl_context context();

	}

}

#endif // OPENCL_OPENCL_HH
