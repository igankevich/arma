#ifndef OPENCL_KERNEL_HH
#define OPENCL_KERNEL_HH

#include "opencl.hh"
#include "buffer.hh"

#include <vector>
#include <string>

namespace arma {

	namespace opencl {

		class Local_mem {
		    size_t bytes;

		public:
		    Local_mem(size_t sz): bytes(sz) {}
		    inline size_t size() const { return bytes; }
		};

		template<class T>
		inline size_t
		get_size_of(const T&) {
		    return sizeof(T);
		}

		template<>
		inline size_t
		get_size_of<Local_mem>(const Local_mem& mem) {
		    return mem.size();
		}

		template<class T>
		inline size_t
		get_size_of(const Buffer<T>& rhs) {
		    return sizeof(cl_mem);
		}

		template<class T>
		inline const void*
		get_addr_of(const T& obj) {
		    return &obj;
		}

		template<>
		inline const void*
		get_addr_of<Local_mem>(const Local_mem&) {
		    return 0;
		}

		template<class T>
		inline const void*
		get_addr_of(const Buffer<T>& rhs) {
		    return rhs.ptr();
		}

		class Kernel {
			typedef std::vector<size_t> size_type;
			cl_kernel _kernel;
			size_type _globalsize;
			size_type _localsize;
			std::string _name;

		public:
			explicit
			Kernel(
				cl_kernel k,
				const size_type& global,
				const size_type& local
			):
			_kernel(k),
			_globalsize(global),
			_localsize(local),
			_name(get_function_name(k))
			{}

			inline explicit
			Kernel(
				const char* name,
				const char* src,
				const size_type& global,
				const size_type& local
			):
			Kernel(get_kernel(name, src), global, local)
			{}

			template<class ... Args>
			inline void
			operator()(const Args& ... args) {
				call(0, args...);
			}

		private:
    		inline void
			call(cl_uint) {
    		    run();
    		}

			template<class T, class ... Args>
			inline void
			call(cl_uint idx, const T& argN, const Args& ... args) {
				check(clSetKernelArg(
					_kernel,
					idx,
					get_size_of(argN),
					get_addr_of(argN)
				), idx);
				call(idx+1, args...);
			}

			static std::string
			get_function_name(cl_kernel kernel);

			void run();

			void check(cl_int err, cl_int arg);

		};

	}

}

#endif // OPENCL_KERNEL_HH
