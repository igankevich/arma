#include "ma_algorithm.hh"

#include <string>
#include <stdexcept>
#include <iostream>

std::istream&
arma::operator>>(std::istream& in, MA_algorithm& rhs) {
	std::string name;
	in >> std::ws >> name;
	if (name == "fixed_point_iteration") {
		rhs = MA_algorithm::Fixed_point_iteration;
	} else if (name == "newton_raphson") {
		rhs = MA_algorithm::Newton_Raphson;
	} else {
		in.setstate(std::ios::failbit);
		std::clog << "Invalid MA algorithm: " << name << std::endl;
		throw std::runtime_error("bad algorithm");
	}
	return in;
}

const char*
arma::to_string(MA_algorithm rhs) {
	switch (rhs) {
		case MA_algorithm::Fixed_point_iteration: return "fixed_point_iteration";
		case MA_algorithm::Newton_Raphson: return "newton_raphson";
		default: return "UNKNOWN";
	}
}

std::ostream&
arma::operator<<(std::ostream& out, const MA_algorithm& rhs) {
	return out << to_string(rhs);
}


