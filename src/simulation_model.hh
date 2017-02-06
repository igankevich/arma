#ifndef SIMULATION_MODEL_HH
#define SIMULATION_MODEL_HH

#include <istream>
#include <ostream>
#include <iostream>
#include <string>
#include <stdexcept>

namespace arma {

	enum struct Simulation_model {
		Autoregressive,
		Moving_average,
		ARMA,
	};

	std::istream& operator>>(std::istream& in, Simulation_model& rhs) {
		std::string name;
		in >> std::ws >> name;
		if (name == "AR") {
			rhs = Simulation_model::Autoregressive;
		} else if (name == "MA") {
			rhs = Simulation_model::Moving_average;
		} else if (name == "ARMA") {
			rhs = Simulation_model::ARMA;
		} else {
			in.setstate(std::ios::failbit);
			std::clog << "Invalid model: " << name << std::endl;
			throw std::runtime_error("bad model");
		}
		return in;
	}

	const char*
	to_string(Simulation_model rhs) {
		switch (rhs) {
			case Simulation_model::Autoregressive:
				return "AR";
			case Simulation_model::Moving_average:
				return "MA";
			case Simulation_model::ARMA:
				return "ARMA";
			default:
				return "UNKNOWN";
		}
	}

	std::ostream& operator<<(std::ostream& out, const Simulation_model& rhs) {
		return out << to_string(rhs);
	}
}

#endif // SIMULATION_MODEL_HH
