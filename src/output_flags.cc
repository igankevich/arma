#include "output_flags.hh"
#include <iostream>
#include <stdexcept>
#include <string>
#include <array>
#include <algorithm>
#include <cassert>

namespace {

	typedef std::pair<arma::Output_flags::Flag,std::string> flag_pair;

	std::array<std::string,9> all_flags{{
		"none",
		"summary",
		"qq",
		"waves",
		"acf",
		"csv",
		"blitz",
		"binary",
		"surface",
	}};

}

std::string
arma::get_filename(const std::string& prefix, Output_flags::Flag flag) {
	std::string f;
	f.append(prefix);
	if (flag == Output_flags::Flag::CSV) {
		f.append(".csv");
	} else if (flag == Output_flags::Flag::Binary) {
		f.append(".bin");
	}
	return f;
}

void
arma::Output_flags::prune() {
	// do nothing if all flags are disabled
	if (isset(Flag::None)) {
		return;
	}
	// set default output format if none is specified
	if (isset(Flag::Surface) && !isset(Flag::Blitz) &&
		!isset(Flag::CSV) && !isset(Flag::Binary))
	{
		setf(Flag::Blitz);
	}
}

std::istream&
arma::operator>>(std::istream& in, Output_flags& rhs) {
	std::string name;
	bool stopped = false;
	while (!stopped && !in.eof()) {
		char ch = in.get();
		if (ch == '\n') {
			stopped = true;
		}
		if (ch == '\n' || ch == ',' || in.eof()) {
			auto result = std::find(
				all_flags.begin(),
				all_flags.end(),
				name
			);
			if (result == all_flags.end()) {
				in.setstate(std::ios::failbit);
				std::cerr << "Invalid output flag: " << name << std::endl;
				throw std::runtime_error("bad output flag");
			}
			Output_flags::Flag f =
				static_cast<Output_flags::Flag>(result - all_flags.begin());
			if (f == Output_flags::None) {
				rhs._flags.reset();
			} else {
				rhs.setf(f);
			}
			name.clear();
		} else if (!std::isspace(ch)) {
			name.push_back(ch);
		}
	}
	rhs.prune();
	return in;
}

std::ostream&
arma::operator<<(std::ostream& out, const Output_flags& rhs) {
	if (rhs._flags.none()) {
		out << "none";
	} else {
		const size_t n = rhs._flags.size();
		bool first = true;
		for (size_t i=1; i<n; ++i) {
			if (rhs._flags.test(i) && i<all_flags.size()) {
				if (first) {
					first = false;
				} else {
					out << ',';
				}
				out << all_flags[i];
			}
		}
	}
	return out;
}

#if ARMA_BSCHEDULER

void
arma::Output_flags
::write(sys::pstream& out) const {
	out << uint64_t(this->_flags.to_ulong());
}

void
arma::Output_flags
::read(sys::pstream& in) {
	uint64_t n = 0;
	in >> n;
	this->_flags = bitset_type(n);
}

#endif
