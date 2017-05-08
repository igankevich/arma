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

		void init();

		cl::Context context();

		cl::CommandQueue command_queue();

		void compile(const char* src);

		cl::Kernel get_kernel(const char* name, const char* src);

		class GL_object_guard {
			std::vector<cl::Memory> _objs;

		public:
			GL_object_guard(cl::Memory mem);
			~GL_object_guard();

		};

	}

}

namespace cl {

	std::ostream&
	operator<<(std::ostream& out, const cl::Error& err);

}


#endif // OPENCL_OPENCL_HH
