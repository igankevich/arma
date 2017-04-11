#include "opencl.hh"
#include "params.hh"
#include "util.hh"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iterator>

namespace {

	#include "get_opencl_error_string.cc"

	void
	check_err(cl_int err, const char* description) {
		if (err != CL_SUCCESS) {
			std::cerr
				<< "OpenCL error " << err << " in " << description << ':'
				<< " error_code=" << get_opencl_error_string(err)
				<< std::endl;
			std::exit(err);
		}
	}

	void
	onError(const char *errinfo, const void*, size_t, void*) {
		std::cerr << "Error from OpenCL. " << errinfo << std::endl;
		std::exit(1);
	}

	class Platform {
		cl_platform_id _id;
		std::string _name;
		std::string _vendor;

	public:
		explicit
		Platform(cl_platform_id id):
		_id(id)
		{
			get_info(CL_PLATFORM_NAME, _name);
			get_info(CL_PLATFORM_VENDOR, _vendor);
		}

		cl_platform_id
		id() const noexcept {
			return _id;
		}

		const std::string
		name() const noexcept {
			return _name;
		}

		const std::string
		vendor() const noexcept {
			return _vendor;
		}

		bool
		contains(const std::string& sub) const noexcept {
			return _name.find(sub) != std::string::npos
				|| _vendor.find(sub) != std::string::npos;
		}

		friend std::ostream&
		operator<<(std::ostream& out, const Platform& rhs) {
			return out
				<< "name=" << rhs._name
				<< ",vendor=" << rhs._vendor;
		}

	private:
		void
		get_info(cl_platform_info info, std::string& str) {
			size_t n = 0;
			clGetPlatformInfo(_id, info, 0, 0, &n);
			if (n > 0) {
				str.resize(n);
				clGetPlatformInfo(_id, info, n, &str[0], 0);
			} else {
				str.clear();
			}
		}
	};

	enum struct Device_type: cl_device_type {
		Default = CL_DEVICE_TYPE_DEFAULT,
		CPU = CL_DEVICE_TYPE_CPU,
		GPU = CL_DEVICE_TYPE_GPU,
		Accelerator = CL_DEVICE_TYPE_ACCELERATOR,
		Custom = CL_DEVICE_TYPE_CUSTOM,
		All = CL_DEVICE_TYPE_ALL
	};

	std::istream&
	operator>>(std::istream& in, Device_type& rhs) {
		std::string name;
		in >> std::ws >> name;
		if (name == "default") {
			rhs = Device_type::Default;
		} else if (name == "CPU") {
			rhs = Device_type::CPU;
		} else if (name == "GPU") {
			rhs = Device_type::GPU;
		} else if (name == "accelerator") {
			rhs = Device_type::Accelerator;
		} else if (name == "custom") {
			rhs = Device_type::Custom;
		} else if (name == "all") {
			rhs = Device_type::All;
		} else {
			in.setstate(std::ios::failbit);
			std::clog << "Invalid device type: " << name << std::endl;
			throw std::runtime_error("bad device type");
		}
		return in;
	}

	const char*
	to_string(Device_type rhs) {
		switch (rhs) {
			case Device_type::Default: return "default";
			case Device_type::CPU: return "CPU";
			case Device_type::GPU: return "GPU";
			case Device_type::Accelerator: return "accelerator";
			case Device_type::Custom: return "custom";
			case Device_type::All: return "all";
			default: return "UNKNOWN";
		}
	}

	std::ostream&
	operator<<(std::ostream& out, const Device_type& rhs) {
		return out << to_string(rhs);
	}

	std::ostream&
	operator<<(std::ostream& out, const std::vector<Platform>& rhs) {
		std::copy(
			rhs.begin(),
			rhs.end(),
			std::ostream_iterator<Platform>(out, "\n")
		);
		return out;
	}

	class OpenCL {
		cl_context _context = nullptr;
		cl_device_id _device = nullptr;
		cl_command_queue _cmdqueue = nullptr;

	public:
		OpenCL() {
			cl_int err = CL_SUCCESS;
			std::string platform_name;
			Device_type device_type = Device_type::Default;
			{
				std::ifstream in("opencl.conf");
				if (in.is_open()) {
					sys::parameter_map params({
						{"platform_name", sys::make_param(platform_name)},
						{"device_type", sys::make_param(device_type)},
					}, false);
					in >> params;
				}
			}
			// TODO add timeout for querying a list of platforms
			unsigned int nplatforms = 0;
			err = clGetPlatformIDs(0, nullptr, &nplatforms);
			if (nplatforms == 0) {
				std::clog << "Unable to find any OpenCL platform. Terminating." << std::endl;
				std::exit(1);
			}
			std::vector<cl_platform_id> ids(nplatforms);
			err = clGetPlatformIDs(nplatforms, ids.data(), &nplatforms);
			std::vector<Platform> platforms;
			std::transform(
				ids.begin(),
				ids.end(),
				std::back_inserter(platforms),
				[] (cl_platform_id id) {
					return Platform(id);
				}
			);
			std::vector<Platform>::iterator result = platforms.begin();
			if (!platform_name.empty()) {
				result = std::find_if(
					platforms.begin(),
					platforms.end(),
					[&platform_name] (const Platform& rhs) {
						return rhs.contains(platform_name);
					}
				);
				if (result == platforms.end()) {
					std::clog << "Unable to find requested OpenCL platform. Platform name = "
						<< platform_name << std::endl;
					std::clog << "Here is the list of all OpenCL platforms:\n";
					std::clog << platforms;
					std::exit(1);
				}
			}
			arma::write_key_value(std::clog, "OpenCL platform", result->name());
			arma::write_key_value(std::clog, "OpenCL platform vendor", result->vendor());
			arma::write_key_value(std::clog, "OpenCL device type", device_type);
			check_err(err, "clGetPlatformIDs");
			cl_context_properties props[] = {
				CL_CONTEXT_PLATFORM, (cl_context_properties) result->id(),
				0
			};
			_context = clCreateContextFromType(
				props,
				cl_device_type(device_type),
				onError,
				0,
				&err
			);
			check_err(err, "clCreateContextFromType");
			err = clGetContextInfo(
				_context,
				CL_CONTEXT_DEVICES,
				sizeof(cl_device_id),
				&_device,
				0
			);
			check_err(err, "clGetContextInfo");
			_cmdqueue = clCreateCommandQueue(_context, _device, 0, &err);
			check_err(err, "clCreateCommandQueue");
		}

		~OpenCL() {
			if (_cmdqueue) {
				clReleaseCommandQueue(_cmdqueue);
			}
			if (_context) {
				clReleaseContext(_context);
			}
		}

		cl_context
		context() const noexcept {
			return _context;
		}

	} __opencl_instance;

}


cl_context
arma::opencl::context() {
	return __opencl_instance.context();
}
