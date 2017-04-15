#include "opencl.hh"
#include "params.hh"
#include "util.hh"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <stdexcept>
#include <exception>

#if ARMA_OPENGL
#include <GL/glx.h>
#include <CL/cl_gl.h>
#endif

namespace cl {

	std::ostream&
	operator<<(std::ostream& out, const cl::Platform& rhs) {
		return out
			<< "name=" << rhs.getInfo<CL_PLATFORM_NAME>()
			<< ",vendor=" << rhs.getInfo<CL_PLATFORM_VENDOR>();
	}

	std::ostream&
	operator<<(std::ostream& out, const cl::Error& err) {
		return out << "ERROR: " << err.what() << '(' << err.err() << ')';
	}

}

namespace {

	void
	trim_right(std::string& rhs) {
		while (!rhs.empty() && rhs.back() <= ' ') { rhs.pop_back(); }
	}

	[[noreturn]] void
	print_error_and_exit(cl::Error err) {
		std::cerr << err << std::endl;
		std::exit(1);
	}

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
	operator<<(std::ostream& out, const std::vector<cl::Platform>& rhs) {
		std::copy(
			rhs.begin(),
			rhs.end(),
			std::ostream_iterator<cl::Platform>(out, "\n")
		);
		return out;
	}

	class OpenCL {
		cl::Context _context;
		std::vector<cl::Device> _devices;
		cl::CommandQueue _cmdqueue;
		std::string _options;
		std::unordered_map<std::string,cl::Kernel> _kernels;

	public:
		void
		init_opencl() {
			try {
				do_init_opencl();
			} catch (cl::Error err) {
				print_error_and_exit(err);
			}
		}

		void
		do_init_opencl() {
			std::string platform_name;
			Device_type device_type = Device_type::Default;
			{
				std::ifstream in("opencl.conf");
				if (in.is_open()) {
					sys::parameter_map params({
						{"platform_name", sys::make_param(platform_name)},
						{"device_type", sys::make_param(device_type)},
						{"options", sys::make_param(_options)},
					}, false);
					in >> params;
				}
				_options += " -DARMA_REAL_TYPE=" ARMA_STRINGIFY(ARMA_REAL_TYPE);
			}
			std::vector<cl::Platform> platforms;
			cl::Platform::get(&platforms);
			if (platforms.empty()) {
				std::clog
					<< "Unable to find any OpenCL platform. Terminating."
					<< std::endl;
				std::exit(1);
			}
			std::vector<cl::Platform>::iterator result = platforms.begin();
			if (!platform_name.empty()) {
				result = std::find_if(
					platforms.begin(),
					platforms.end(),
					[&platform_name] (const cl::Platform& rhs) {
						std::string name = rhs.getInfo<CL_PLATFORM_NAME>();
						std::string vendor = rhs.getInfo<CL_PLATFORM_VENDOR>();
						return name.find(platform_name) != std::string::npos
							|| vendor.find(platform_name) != std::string::npos;
					}
				);
				if (result == platforms.end()) {
					std::clog << "Unable to find requested OpenCL platform. "
						"Platform name = " << platform_name << std::endl;
					std::clog << "Here is the list of all OpenCL platforms:\n";
					std::clog << platforms;
					std::exit(1);
				}
			}
			arma::write_key_value(
				std::clog,
				"OpenCL platform",
				result->getInfo<CL_PLATFORM_NAME>()
			);
			arma::write_key_value(
				std::clog,
				"OpenCL platform vendor",
				result->getInfo<CL_PLATFORM_VENDOR>()
			);
			#if ARMA_OPENGL
			GLXContext glx_context = ::glXGetCurrentContext();
			Display* glx_display = ::glXGetCurrentDisplay();
			if (!glx_context || !glx_display) {
				std::clog << "Unable to get current GLX display or context. "
					"Please, initialise OpenGL before OpenCL. "
					"Terminating." << std::endl;
				std::exit(1);
			}
			#endif
			cl_context_properties props[] = {
				CL_CONTEXT_PLATFORM, (cl_context_properties) (*result)(),
				#if ARMA_OPENGL
				CL_GL_CONTEXT_KHR, (cl_context_properties) glx_context,
				CL_GLX_DISPLAY_KHR, (cl_context_properties) glx_display,
				#endif
				0
			};
			_context = cl::Context(cl_device_type(device_type), props);
			_devices = _context.getInfo<CL_CONTEXT_DEVICES>();
			const std::string extensions = _devices[0].getInfo<CL_DEVICE_EXTENSIONS>();
			if (extensions.find("cl_khr_gl_sharing") == std::string::npos) {
				std::clog << "OpenCL and OpenGL context sharing (cl_khr_gl_sharing) "
					"is not supported. Terminating." << std::endl;
				std::exit(1);
			}
			cl_command_queue_properties qprops = 0;
			auto supported = _devices[0].getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
			if (supported & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
				qprops |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
			}
			arma::write_key_value(
				std::clog,
				"OpenCL device type",
				Device_type(_devices[0].getInfo<CL_DEVICE_TYPE>())
			);
			#if ARMA_PROFILE
			qprops |= CL_QUEUE_PROFILING_ENABLE;
			#endif
			_cmdqueue = cl::CommandQueue(_context, _devices[0], qprops);
		}

		cl::Context
		context() const noexcept {
			return _context;
		}

		cl::CommandQueue
		command_queue() const noexcept {
			return _cmdqueue;
		}

		void
		compile(const char* src) {
			cl::Program prg = new_program(src);
			createKernels(prg);
		}

		cl_kernel
		get_kernel(const std::string& name) const {
		    auto it = _kernels.find(name);
		    if (it == _kernels.end()) {
				return nullptr;
		    }
		    return it->second();
		}

	private:

		cl::Program
		new_program(const char* src) {
			cl::Program program(_context, src);
			try {
				program.build({_devices[0]}, _options.data());
			} catch (cl::Error err) {
				if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
					std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
						_devices[0]
					);
					std::clog << log;
					std::exit(1);
				} else {
					print_error_and_exit(err);
				}
			}
			return program;
		}

		void
		createKernels(cl::Program program) {
			std::vector<cl::Kernel> all_kernels;
			program.createKernels(&all_kernels);
			for (const cl::Kernel& kernel : all_kernels) {
				std::string name;
				kernel.getInfo(CL_KERNEL_FUNCTION_NAME, &name);
				trim_right(name);
				_kernels.emplace(name, kernel);
			}
		}


	} __opencl_instance;

}


cl::Context
arma::opencl::context() {
	return __opencl_instance.context();
}

cl::CommandQueue
arma::opencl::command_queue() {
	return __opencl_instance.command_queue();
}

void
arma::opencl::compile(const char* src) {
	__opencl_instance.compile(src);
}

cl::Kernel
arma::opencl::get_kernel(const char* name, const char* src) {
	cl_kernel kernel = __opencl_instance.get_kernel(name);
	if (!kernel) {
		compile(src);
	}
	kernel = __opencl_instance.get_kernel(name);
	if (!kernel) {
		std::cerr << "OpenCL kernel not found: " << name << std::endl;
		throw std::runtime_error("bad kernel");
	}
	return cl::Kernel(kernel);
}

void
arma::opencl::init() {
	__opencl_instance.init_opencl();
}
