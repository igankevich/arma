#ifndef OPENCL_DEVICE_TYPE_HH
#define OPENCL_DEVICE_TYPE_HH

#include <istream>
#include <ostream>
#include "cl.hh"

namespace arma {

	namespace opencl {

		enum struct Device_type: cl_device_type {
			Default = CL_DEVICE_TYPE_DEFAULT,
			CPU = CL_DEVICE_TYPE_CPU,
			GPU = CL_DEVICE_TYPE_GPU,
			Accelerator = CL_DEVICE_TYPE_ACCELERATOR,
			Custom = CL_DEVICE_TYPE_CUSTOM,
			All = CL_DEVICE_TYPE_ALL
		};

		std::istream&
		operator>>(std::istream& in, Device_type& rhs);

		const char*
		to_string(Device_type rhs);

		inline std::ostream&
		operator<<(std::ostream& out, const Device_type& rhs) {
			return out << to_string(rhs);
		}

	}

}

#endif // OPENCL_DEVICE_TYPE_HH
