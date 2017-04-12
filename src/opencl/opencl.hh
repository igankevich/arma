#ifndef OPENCL_OPENCL_HH
#define OPENCL_OPENCL_HH

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#define ARMA_STRINGIFY_IMPL(x) #x
#define ARMA_STRINGIFY(x) ARMA_STRINGIFY_IMPL(x)

namespace arma {

	namespace opencl {

		cl_context context();

		cl_command_queue command_queue();

		void compile(const char* src);

		cl_kernel get_kernel(const char* name, const char* src);

		void check_err(cl_int err, const char* description);

	}

}

#endif // OPENCL_OPENCL_HH
