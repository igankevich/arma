#include "params.hh"

#include <limits>
#include <iostream>
#include <iomanip>

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

	template<char CH>
	struct const_char {
		friend std::istream&
		operator>>(std::istream& in, const const_char& rhs) {
			char ch;
			if ((ch = in.get()) != CH) {
				in.putback(ch);
				in.setstate(std::ios::failbit);
			}
			return in;
		}
	};

	void
	trim_right(std::string& rhs) {
		while (!rhs.empty() && rhs.back() <= ' ') { rhs.pop_back(); }
	}

}

std::istream&
sys::operator>>(std::istream& in, parameter_map& rhs) {
	if (rhs._parens && !(in >> std::ws >> const_char<'{'>())) {
		std::clog << "Expecting \"{\"." << std::endl;
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
			std::clog << "Unknown parameter: " << name << std::endl;
			in.setstate(std::ios::failbit);
		} else {
			result->second(in, name.data());
		}
	}
	return in;
}

