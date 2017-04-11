#ifndef OPENCL_OPENCL_HH
#define OPENCL_OPENCL_HH

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#define ARMA_STRINGIFY_IMPL(x) #x
#define ARMA_STRINGIFY(x) ARMA_STRINGIFY_IMPL(x)

namespace arma {

	namespace opencl {

		cl_context context();

		void compile(const char* src);

		cl_kernel get_kernel(const char* name);

	}

}

#endif // OPENCL_OPENCL_HH
