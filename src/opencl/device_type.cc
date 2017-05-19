#include "device_type.hh"

#include <string>
#include <iomanip>
#include <iostream>
#include <stdexcept>

std::istream&
arma::opencl::operator>>(std::istream& in, Device_type& rhs) {
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
		std::cerr << "Invalid device type: " << name << std::endl;
		throw std::runtime_error("bad device type");
	}
	return in;
}

const char*
arma::opencl::to_string(Device_type rhs) {
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

