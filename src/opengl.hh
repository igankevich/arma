#ifndef OPENGL_HH
#define OPENGL_HH

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glx.h>
#include <type_traits>

namespace arma {

	/// \brief OpenGL glue code.
	namespace opengl {

		/**
		\brief Maps `GL*` types to the corresponding `GLenum` constants
		*/
		template<class T>
		struct GL_type;

		#define MAKE_GL_TYPE(x, y) \
			template<> \
			struct GL_type<x>: public std::integral_constant<GLenum, y> {};

		MAKE_GL_TYPE(GLfloat, GL_FLOAT);
		MAKE_GL_TYPE(GLdouble, GL_DOUBLE);
		MAKE_GL_TYPE(GLbyte, GL_BYTE);
		MAKE_GL_TYPE(GLubyte, GL_UNSIGNED_BYTE);
		MAKE_GL_TYPE(GLshort, GL_SHORT);
		MAKE_GL_TYPE(GLushort, GL_UNSIGNED_SHORT);
		MAKE_GL_TYPE(GLint, GL_INT);
		MAKE_GL_TYPE(GLuint, GL_UNSIGNED_INT);

		#undef MAKE_GL_TYPE

	}

}


#endif // OPENGL_HH
