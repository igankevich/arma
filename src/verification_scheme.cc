#include "verification_scheme.hh"
#include <iostream>
#include <stdexcept>
#include <string>

std::istream&
arma::operator>>(std::istream& in, Verification_scheme& rhs) {
	std::string name;
	in >> std::ws >> name;
	if (name == "none") {
		rhs = Verification_scheme::No_verification;
	} else if (name == "summary") {
		rhs = Verification_scheme::Summary;
	} else if (name == "quantile") {
		rhs = Verification_scheme::Quantile;
	} else if (name == "manual") {
		rhs = Verification_scheme::Manual;
	} else {
		in.setstate(std::ios::failbit);
		std::cerr << "Invalid verification scheme: " << name << std::endl;
		throw std::runtime_error("bad varification scheme");
	}
	return in;
}

const char*
arma::to_string(Verification_scheme rhs) {
	switch (rhs) {
		case Verification_scheme::No_verification: return "none";
		case Verification_scheme::Summary: return "summary";
		case Verification_scheme::Quantile: return "quantile";
		case Verification_scheme::Manual: return "manual";
		default: return "UNKNOWN";
	}
}


std::ostream&
arma::operator<<(std::ostream& out, const Verification_scheme& rhs) {
	return out << to_string(rhs);
}


