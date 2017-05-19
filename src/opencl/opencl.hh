#ifndef OPENCL_OPENCL_HH
#define OPENCL_OPENCL_HH

#include "cl.hh"

namespace arma {

	/**
	\brief OpenCL glue code.
	*/
	namespace opencl {

		void init();

		cl::Context context();

		const std::vector<cl::Device>& devices();

		cl::CommandQueue command_queue();

		void compile(const char* src);

		cl::Kernel get_kernel(const char* name, const char* src);
		cl::Kernel get_kernel(const char* name);

		bool
		supports_gl_sharing(cl::Device dev);

	}

}

namespace cl {

	std::ostream&
	operator<<(std::ostream& out, const cl::Error& err);

}


#endif // OPENCL_OPENCL_HH
