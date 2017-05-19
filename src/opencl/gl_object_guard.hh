#ifndef OPENCL_GL_OBJECT_GUARD_HH
#define OPENCL_GL_OBJECT_GUARD_HH

#include <initializer_list>
#include "opencl.hh"

namespace arma {

	namespace opencl {

		/**
		\brief An object that manages shared OpenCL/OpenGL objects
		by keeping them acquired.

		On construction it acquires shared OpenGL buffer by calling
		<a href="https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clEnqueueAcquireGLObjects.html">clEnqueueAcquireGLObjects</a>,
		on destruction it releases the buffer by calling
		<a href="https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clEnqueueReleaseGLObjects.html">clEnqueueReleaseGLObjects</a>.
		If OpenCL/OpenGL sharing is not supported, then no calls are made.
		*/
		class GL_object_guard {
			std::vector<cl::Memory> _objs;
			bool _glsharing;

		public:
			explicit
			GL_object_guard(cl::Memory mem);
			GL_object_guard(std::initializer_list<cl::Memory> mem);
			~GL_object_guard();

			GL_object_guard() = delete;
			GL_object_guard(const GL_object_guard&) = delete;
			GL_object_guard(GL_object_guard&&) = delete;
			GL_object_guard& operator=(const GL_object_guard&) = delete;
		};

	}

}

#endif // OPENCL_GL_OBJECT_GUARD_HH
