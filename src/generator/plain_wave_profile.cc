#include "plain_wave_profile.hh"

#include <iostream>
#include <istream>

std::istream&
arma::generator::operator>>(std::istream& in, Plain_wave_profile& rhs) {
	std::string name;
	in >> std::ws >> name;
	if (name == "sin") {
		rhs = Plain_wave_profile::Sine;
	} else if (name == "cos") {
		rhs = Plain_wave_profile::Cosine;
	} else if (name == "stokes") {
		rhs = Plain_wave_profile::Stokes;
	} else if (name == "standing_wave") {
		rhs = Plain_wave_profile::Standing_wave;
	} else {
		in.setstate(std::ios::failbit);
		std::cerr << "Invalid plain wave function: " << name << std::endl;
		throw std::runtime_error("bad function");
	}
	return in;
}

const char*
arma::generator::to_string(Plain_wave_profile rhs) {
	switch (rhs) {
		case Plain_wave_profile::Sine: return "sin";
		case Plain_wave_profile::Cosine: return "cos";
		case Plain_wave_profile::Stokes: return "stokes";
		case Plain_wave_profile::Standing_wave: return "standing_wave";
		default: return "UNKNOWN";
	}
}
