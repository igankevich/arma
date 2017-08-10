#ifndef OPENCL_ARRAY_HH
#define OPENCL_ARRAY_HH

#include <stdexcept>
#include <blitz/array.h>
#include "opencl.hh"

namespace arma {

	namespace bits {

		inline cl::NDRange
		make_ndrange(const blitz::TinyVector<int,0>& rhs) noexcept {
			return cl::NDRange();
		}

		inline cl::NDRange
		make_ndrange(const blitz::TinyVector<int,1>& rhs) noexcept {
			return cl::NDRange(rhs(0));
		}

		inline cl::NDRange
		make_ndrange(const blitz::TinyVector<int,2>& rhs) noexcept {
			return cl::NDRange(rhs(0), rhs(1));
		}

		inline cl::NDRange
		make_ndrange(const blitz::TinyVector<int,3>& rhs) noexcept {
			return cl::NDRange(rhs(0), rhs(1), rhs(2));
		}

	}

	namespace opencl {

		template <class T, int N>
		class Array: public blitz::Array<T,N> {

		public:
			typedef blitz::Array<T,N> base_type;
			typedef blitz::TinyVector<int,N> shape_type;
			typedef blitz::RectDomain<N> rect_domain_type;

		public:
			using base_type::operator=;
			using base_type::operator();

		private:
			cl::Buffer _buffer;

		public:

			Array() = default;
			Array(const Array&) = default;
			~Array() = default;

			Array(Array&& rhs):
			base_type(std::forward<base_type>(rhs)),
			_buffer(std::forward<cl::Buffer>(rhs._buffer))
			{}

			template <class Expr>
			explicit
			Array(Expr rhs):
			base_type(rhs)
			{}

			Array(const base_type& rhs):
			base_type(rhs)
			{}

			explicit
			Array(const shape_type& rhs):
			base_type(rhs)
			{}

			explicit
			Array(const rect_domain_type& rhs):
			base_type(rhs)
			{}

			inline Array&
			operator=(const Array& rhs) {
				this->_buffer = rhs._buffer;
				this->base_type::operator=(static_cast<const base_type&>(rhs));
				return *this;
			}

			inline Array&
			operator=(const base_type& rhs) {
				this->base_type::operator=(static_cast<const base_type&>(rhs));
				return *this;
			}

			inline Array
			copy() const {
				return Array(this->base_type::copy());
			}

			inline void
			reference(const Array& rhs) {
				this->base_type::reference(rhs);
				this->_buffer = rhs._buffer;
			}

			inline T*
			data_begin() noexcept {
				return this->data();
			}

			inline T*
			data_end() noexcept {
				return this->data() + this->numElements();
			}

			inline cl::Buffer
			buffer() const noexcept {
				return this->_buffer;
			}

			inline void
			copy_to_device(cl_mem_flags flags=CL_MEM_READ_WRITE) {
				if (this->_buffer()) {
					cl::copy(
						command_queue(),
						this->data_begin(),
						this->data_end(),
						this->_buffer
					);
				} else {
					this->_buffer = cl::Buffer(
						context(),
						this->data_begin(),
						this->data_end(),
						flags & CL_MEM_READ_ONLY,
						flags & CL_MEM_USE_HOST_PTR
					);
				}
			}

			inline void
			init_on_device(cl_mem_flags flags=CL_MEM_READ_WRITE) {
				if (this->_buffer()) {
					throw std::runtime_error("buffer already initialised");
				}
				this->_buffer = cl::Buffer(
					context(),
					flags,
					this->numElements()*sizeof(T)
				);
			}

			inline void
			copy_to_host() {
				if (!this->_buffer()) {
					throw std::runtime_error("uninitialised buffer");
				}
				cl::copy(
					command_queue(),
					this->_buffer,
					this->data_begin(),
					this->data_end()
				);
			}

			inline void
			copy_to_host_if_exists() {
				if (this->_buffer()) {
					this->copy_to_host();
				}
			}

			inline cl::NDRange
			ndrange() const noexcept {
				return ::arma::bits::make_ndrange(this->shape());
			}

			inline void
			compute(cl::Kernel kernel) {
				command_queue().enqueueNDRangeKernel(
					kernel,
					cl::NullRange,
					this->ndrange()
				);
			}

		};

	}

}

#endif // OPENCL_ARRAY_HH
