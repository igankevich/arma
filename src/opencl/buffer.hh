#ifndef OPENCL_BUFFER_HH
#define OPENCL_BUFFER_HH

#include "opencl.hh"

namespace arma {

	namespace opencl {

		template <class T>
		class Buffer {
			cl_mem _buffer = nullptr;
			size_t _size = 0;

		public:
			inline explicit
			Buffer(size_t n, cl_mem_flags flags):
			_buffer(nullptr),
			_size(n)
			{
				cl_int err = CL_SUCCESS;
				_buffer = clCreateBuffer(context(), flags, n*sizeof(T), 0, &err);
				check_err(err, "clCreateBuffer");
			}

			inline explicit
			Buffer(const T* data, size_t n, cl_mem_flags flags):
			Buffer(n, flags)
			{ copy_in(data, n); }

			inline
			~Buffer() {
				clReleaseMemObject(_buffer);
			}

			inline void
			copy_in(const T* data, size_t n) {
				cl_int err = clEnqueueWriteBuffer(
					command_queue(),
					_buffer,
					CL_TRUE,
					0, n*sizeof(T),
					data,
					0, 0, 0
				);
				check_err(err, "clEnqueueWriteBuffer");
			}

			inline void
			copy_out(T* data, size_t n) const {
				cl_int err = clEnqueueReadBuffer(
					command_queue(),
					_buffer,
					CL_TRUE,
					0, n*sizeof(T),
					data,
					0, 0, 0
				);
				check_err(err, "clEnqueueReadBuffer");
			}

			inline const cl_mem*
			ptr() const noexcept {
				return &_buffer;
			}

			inline size_t
			nbytes() const noexcept {
				return _size*sizeof(T);
			}
		};

	}

}

#endif // OPENCL_BUFFER_HH
