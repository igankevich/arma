#include "ar_algorithm.hh"

#include <string>
#include <stdexcept>
#include <iostream>

std::istream&
arma::operator>>(std::istream& in, AR_algorithm& rhs) {
	std::string name;
	in >> std::ws >> name;
	if (name == "gauss_elimination") {
		rhs = AR_algorithm::Gauss_elimination;
	} else if (name == "choi_recursive") {
		rhs = AR_algorithm::Choi;
	} else {
		in.setstate(std::ios::failbit);
		std::clog << "Invalid AR algorithm: " << name << std::endl;
		throw std::runtime_error("bad algorithm");
	}
	return in;
}

const char*
arma::to_string(AR_algorithm rhs) {
	switch (rhs) {
		case AR_algorithm::Gauss_elimination: return "gauss_elimination";
		case AR_algorithm::Choi: return "choi_recursive";
		default: return "UNKNOWN";
	}
}

std::ostream&
arma::operator<<(std::ostream& out, const AR_algorithm& rhs) {
	return out << to_string(rhs);
}


