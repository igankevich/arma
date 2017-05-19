#include "gl_object_guard.hh"

arma::opencl::GL_object_guard::GL_object_guard(cl::Memory mem):
GL_object_guard({mem})
{}

arma::opencl::GL_object_guard::GL_object_guard(
	std::initializer_list<cl::Memory> mem
):
_objs(mem),
_glsharing(supports_gl_sharing(devices()[0]))
{
	if (_glsharing) {
		command_queue().enqueueAcquireGLObjects(&_objs);
	}
}

arma::opencl::GL_object_guard::~GL_object_guard() {
	if (_glsharing) {
		command_queue().enqueueReleaseGLObjects(&_objs);
	}
}

