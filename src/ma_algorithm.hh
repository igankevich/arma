#ifndef MA_ALGORITHM_HH
#define MA_ALGORITHM_HH

#include <istream>
#include <ostream>
#include <iostream>
#include <string>
#include <stdexcept>

namespace arma {

	enum struct MA_algorithm {
		Fixed_point_iteration = 0,
		Newton_Raphson = 1,
	};

	std::istream&
	operator>>(std::istream& in, MA_algorithm& rhs) {
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
	to_string(MA_algorithm rhs) {
		switch (rhs) {
			case MA_algorithm::Fixed_point_iteration: return "fixed_point_iteration";
			case MA_algorithm::Newton_Raphson: return "newton_raphson";
			default: return "UNKNOWN";
		}
	}

	std::ostream&
	operator<<(std::ostream& out, const MA_algorithm& rhs) {
		return out << to_string(rhs);
	}


}

#endif // MA_ALGORITHM_HH
