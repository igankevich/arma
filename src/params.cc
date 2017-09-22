#include "params.hh"

#include <limits>
#include <iostream>
#include <iomanip>

#include "bits/const_char.hh"

namespace {

	/// ignore lines starting with "#"
	std::istream&
	comment(std::istream& in) {
		char ch = 0;
		while (in >> std::ws >> ch && ch == '#') {
			in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			ch = 0;
		}
		if (ch != 0) { in.putback(ch); }
		return in;
	}

	void
	trim_right(std::string& rhs) {
		while (!rhs.empty() && rhs.back() <= ' ') { rhs.pop_back(); }
	}

}

std::istream&
sys::operator>>(std::istream& in, parameter_map& rhs) {
	using arma::bits::const_char;
	if (rhs._parens && !(in >> std::ws >> const_char<'{'>())) {
		std::cerr << "Expecting \"{\"." << std::endl;
	}
	std::string name;
	while (in >> comment) {
		if (rhs._parens) {
			if (in >> std::ws >> const_char<'}'>()) {
				break;
			}
			in.clear();
		}
		if (!(in >> std::ws && std::getline(in, name, '='))) {
			break;
		}
		trim_right(name);
		auto result = rhs._params.find(name);
		if (result == rhs._params.end()) {
			std::cerr << "Unknown parameter: \"" << name << "\"." << std::endl;
			in.setstate(std::ios::failbit);
		} else {
			result->second(in, name.data());
		}
	}
	return in;
}

std::ostream&
sys::operator<<(std::ostream& out, const parameter_map& rhs) {
	if (rhs._parens) {
		out << '{';
	}
//	for (const auto& pair : rhs._params) {
//		out << pair.first << '=' << pair.second << '\n';
//	}
	if (rhs._parens) {
		out << '}';
	}
	return out;
}
