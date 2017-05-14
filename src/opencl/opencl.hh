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

		const std::vector<cl::Device>& devices();

		cl::CommandQueue command_queue();

		void compile(const char* src);

		cl::Kernel get_kernel(const char* name, const char* src);

		bool
		supports_gl_sharing(cl::Device dev);

		class GL_object_guard {
			std::vector<cl::Memory> _objs;
			bool _glsharing;

		public:
			explicit
			GL_object_guard(cl::Memory mem);
			~GL_object_guard();

			GL_object_guard() = delete;
			GL_object_guard(const GL_object_guard&) = delete;
			GL_object_guard(GL_object_guard&&) = delete;
			GL_object_guard& operator=(const GL_object_guard&) = delete;
		};

	}

}

namespace cl {

	std::ostream&
	operator<<(std::ostream& out, const cl::Error& err);

}


#endif // OPENCL_OPENCL_HH
