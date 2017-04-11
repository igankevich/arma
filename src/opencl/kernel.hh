#ifndef OPENCL_KERNEL_HH
#define OPENCL_KERNEL_HH

#include "opencl.hh"
#include <vector>

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
		inline const void*
		get_addr_of(const T& obj) {
		    return &obj;
		}

		template<>
		inline const void*
		get_addr_of<Local_mem>(const Local_mem&) {
		    return 0;
		}

		class Kernel {
			typedef std::vector<size_t> size_type;
			size_type _globalsize;
			size_type _globalsize;
			cl_kernel _kernel;

		public:
			explicit
			Kernel(
				cl_kernel k,
				const size_type& global,
				const size_type& local
			):
			_kernel(k),
			_globalsize(global),
			_localsize(local)
			{}

			explicit
			Kernel(
				const char* name,
				const char* src,
				const size_type& global,
				const size_type& local
			):
			_kernel(get_kernel(name, src)),
			_globalsize(global),
			_localsize(local)
			{}

    		template<class T>
    		inline void operator()(const T& arg0) {
    		    check(clSetKernelArg(_kernel, 0, get_size_of(arg0), get_addr_of(arg0)), 0);
    		    run();
    		}

			template<class T, class ... Args>
			inline void
			operator()(const T& argN, const Args& ... args) const {
				check(clSetKernelArg(kernel, sizeof...(args), get_size_of(args), get_addr_of(args)), sizeof...(args));
				operator()(args...);
			}

		private:

			void
			run() {
				cl_event evt;
				err |= clEnqueueNDRangeKernel(commandQueue, kernel, m, 0, &global_size[0], &local_size[0], 0, 0, &evt);
				err |= clWaitForEvents(1, &evt);
				err |= clReleaseEvent(evt);
				char funcName2[100];
				err |= clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 100, funcName2, 0);
				check_err(err, funcName2);
			}

			void check(cl_int err, cl_int arg) {
				if (err != CL_SUCCESS) {
					char funcName[4096];
					err |= clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(funcName), funcName, 0);
					std::cerr << "OpenCL error=" << err << ", kernel=" << funcName << ", argument=" << arg << std::endl;
					std::exit(err);
				}
			}

		};

	}

}

#endif // OPENCL_KERNEL_HH
