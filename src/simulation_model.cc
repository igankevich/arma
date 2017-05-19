#include "simulation_model.hh"

#include <iostream>
#include <string>
#include <stdexcept>

std::istream&
arma::operator>>(std::istream& in, Simulation_model& rhs) {
	std::string name;
	in >> std::ws >> name;
	if (name == "AR") {
		rhs = Simulation_model::Autoregressive;
	} else if (name == "MA") {
		rhs = Simulation_model::Moving_average;
	} else if (name == "ARMA") {
		rhs = Simulation_model::ARMA;
	} else if (name == "LH") {
		rhs = Simulation_model::Longuet_Higgins;
	} else if (name == "plain_wave") {
		rhs = Simulation_model::Plain_wave_model;
	} else {
		in.setstate(std::ios::failbit);
		std::cerr << "Invalid model: " << name << std::endl;
		throw std::runtime_error("bad model");
	}
	return in;
}

const char*
arma::to_string(Simulation_model rhs) {
	switch (rhs) {
		case Simulation_model::Autoregressive: return "AR";
		case Simulation_model::Moving_average: return "MA";
		case Simulation_model::ARMA: return "ARMA";
		case Simulation_model::Longuet_Higgins: return "LH";
		case Simulation_model::Plain_wave_model: return "plain_wave";
		default: return "UNKNOWN";
	}
}

std::ostream&
arma::operator<<(std::ostream& out, const Simulation_model& rhs) {
	return out << to_string(rhs);
}
