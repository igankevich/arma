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
#include <memory>
#include <functional>

#if ARMA_OPENGL
#include "opengl.hh"
#include <CL/cl_gl.h>
#endif

#include <unistd.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "config.hh"

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
		cl::Platform _platform;
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
				_options += " -I";
				_options += ARMA_OPENCL_SRC_DIR;
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
			_platform = *result;
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
			try {
				_context = cl::Context(cl_device_type(device_type), props);
			} catch (cl::Error err) {
				cl_context_properties newprops[] = {
					CL_CONTEXT_PLATFORM, (cl_context_properties) (*result)(),
					0
				};
				_context = cl::Context(cl_device_type(device_type), newprops);
			}
			_devices = _context.getInfo<CL_CONTEXT_DEVICES>();
			const std::string extensions = _devices[0].getInfo<CL_DEVICE_EXTENSIONS>();
			if (extensions.find("cl_khr_gl_sharing") == std::string::npos) {
				std::clog << "WARNING: "
					"OpenCL and OpenGL context sharing (cl_khr_gl_sharing) "
					"is not supported." << std::endl;
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

		const std::vector<cl::Device>&
		devices() const noexcept {
			return _devices;
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

		std::string
		cache_directory() {
			std::string result;
			const char* xdg_cache_home = std::getenv("XDG_CACHE_HOME");
			if (xdg_cache_home) {
				result = xdg_cache_home;
			} else {
				const char* home = std::getenv("HOME");
				if (!home) {
					struct ::passwd* pwd = ::getpwuid(::getuid());
					if (pwd) {
						home = pwd->pw_dir;
					}
				}
				if (!home) {
					std::clog
						<< "Can not determine neither XDG_CACHE_HOME nor HOME directory. "
						"Setting OpenCL cache to /tmp."
						<< std::endl;
					result = "/tmp";
				} else {
					result.append(home);
					result.append("/.cache");
				}
			}
			result.append("/arma");
			return result;
		}

		void
		init_cache_directory(const std::string& cachedir) {
			if (::mkdir(cachedir.data(), 0755) == -1 && errno != EEXIST) {
				std::cerr << "Unable to create cache directory: " << cachedir << std::endl;
				std::exit(1);
			}
		}

		std::string
		get_binary_filename(cl::Device dev, std::string src) {
			std::string cachedir = cache_directory();
			std::hash<std::string> strhash;
			std::stringstream filename;
			filename
				<< cachedir
				<< '/'
				<< std::hex
				<< std::setw(16) << std::setfill('0') << strhash(_platform.getInfo<CL_PLATFORM_NAME>())
				<< '-'
				<< std::setw(16) << std::setfill('0') << strhash(_platform.getInfo<CL_PLATFORM_VENDOR>())
				<< '-'
				<< std::setw(16) << std::setfill('0') << strhash(dev.getInfo<CL_DEVICE_NAME>())
				<< '-'
				<< std::setw(16) << std::setfill('0') << strhash(src);
			return filename.str();
		}

		void
		cache_binary(cl::Program prg, std::string src) {
			cl_uint ndevices = 0;
			prg.getInfo(CL_PROGRAM_NUM_DEVICES, &ndevices);
			std::vector<size_t> binary_sizes(ndevices);
			prg.getInfo(CL_PROGRAM_BINARY_SIZES, binary_sizes.data());
			std::unique_ptr<unsigned char*,std::function<void(unsigned char**)>>
			binaries(
				new unsigned char*[ndevices],
				[ndevices] (unsigned char** rhs) {
					for (cl_uint i=0; i<ndevices; ++i) {
						delete[] rhs[i];
					}
					delete[] rhs;
				}
			);
			for (cl_uint i=0; i<ndevices; ++i) {
				binaries.get()[i] = new unsigned char[binary_sizes[i]];
			}
			prg.getInfo(CL_PROGRAM_BINARIES, binaries.get());
			std::string cachedir = cache_directory();
			init_cache_directory(cachedir);
			for (cl_uint i=0; i<ndevices; ++i) {
				unsigned char* binary = binaries.get()[i];
				size_t binary_size = binary_sizes[i];
				std::string fname = get_binary_filename(_devices[i], src);
				std::clog << "Cache OpenCL kernel " << fname << std::endl;
				std::ofstream out(fname, std::ios::binary | std::ios::out | std::ios::trunc);
				out.write(reinterpret_cast<const char*>(binary), binary_size);
			}
		}

		std::pair<const void*,size_t>
		read_binary(const std::string& fname) {
			std::ifstream in(fname, std::ios::binary | std::ios::in);
			in.seekg(0, std::ios::end);
			const size_t binary_size = in.tellg();
			char* binary = new char[binary_size];
			in.seekg(0, std::ios::beg);
			in.read(binary, binary_size);
			return std::make_pair(binary, binary_size);
		}

		cl::Program
		new_program(const char* src0) {
			std::string src(src0);
			std::string fname = get_binary_filename(_devices[0], src);
			std::filebuf buf;
			buf.open(fname, std::ios::in);
			cl::Program program;
			if (!buf.is_open()) {
				program = cl::Program(_context, src);
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
				cache_binary(program, src);
			} else {
				std::clog << "Reuse OpenCL kernel " << fname << std::endl;
				cl::Program::Binaries binaries;
				binaries.emplace_back(read_binary(fname));
				program = cl::Program(_context, {_devices[0]}, binaries);
				program.build({_devices[0]}, _options.data());
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

cl::Kernel
arma::opencl::get_kernel(const char* name) {
	std::string path = ARMA_OPENCL_SRC_DIR;
	path += '/';
	path += name;
	path += ".cl";
	std::ifstream in(path);
	if (!in.is_open()) {
		std::cerr << "Unable to load OpenCL kernel " << name
			<< " from " << path << std::endl;
		throw std::runtime_error("bad kernel file");
	}
	std::stringstream str;
	str << in.rdbuf();
	std::string buf(str.str());
	return get_kernel(name, buf.data());
}

void
arma::opencl::init() {
	__opencl_instance.init_opencl();
}

arma::opencl::GL_object_guard::GL_object_guard(cl::Memory mem) {
	_glsharing = supports_gl_sharing(devices()[0]);
	_objs.emplace_back(mem);
	if (_glsharing) {
		command_queue().enqueueAcquireGLObjects(&_objs);
	}
}

arma::opencl::GL_object_guard::~GL_object_guard() {
	if (_glsharing) {
		command_queue().enqueueReleaseGLObjects(&_objs);
	}
}

const std::vector<cl::Device>&
arma::opencl::devices() {
	return __opencl_instance.devices();
}

bool
arma::opencl::supports_gl_sharing(cl::Device dev) {
	const std::string extensions = dev.getInfo<CL_DEVICE_EXTENSIONS>();
	return extensions.find("cl_khr_gl_sharing") != std::string::npos;
}
