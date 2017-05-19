#ifndef OPENCL_VEC_HH
#define OPENCL_VEC_HH

#include <CL/cl.h>

namespace arma {

	namespace opencl {

		namespace bits {

			template <class T, int N>
			struct make_vector {};

			#define MAKE_VECTOR(tp, n) \
				template <> \
				struct make_vector<tp,n> { typedef cl_##tp##n type; };

			MAKE_VECTOR(double, 2);
			MAKE_VECTOR(double, 3);
			MAKE_VECTOR(double, 4);
			MAKE_VECTOR(double, 8);
			MAKE_VECTOR(double, 16);

			MAKE_VECTOR(float, 2);
			MAKE_VECTOR(float, 3);
			MAKE_VECTOR(float, 4);
			MAKE_VECTOR(float, 8);
			MAKE_VECTOR(float, 16);

			MAKE_VECTOR(int, 2);
			MAKE_VECTOR(int, 3);
			MAKE_VECTOR(int, 4);
			MAKE_VECTOR(int, 8);
			MAKE_VECTOR(int, 16);

			#undef MAKE_VECTOR

		}

		/**
		\brief A union that defines a mapping between OpenCL
		device and host types with proper alignment.
		*/
		template <class X, class T, int N>
		union Vec {
			Vec(const X& rhs): _vec1(rhs) {}
			~Vec() {}
			X _vec1;
			typename bits::make_vector<T,N>::type _vec2;
		};

	}

}

#endif // OPENCL_VEC_HH

