#ifndef UTIL_HH
#define UTIL_HH

#include <ostream>
#include <ios>
#include <iomanip>

namespace arma {

	template <class T>
	void
	write_key_value(std::ostream& out, const char* key, const T& value) {
		std::ios::fmtflags oldf =
		    out.setf(std::ios::left, std::ios::adjustfield);
		out << std::setw(30) << key << " = " << value << std::endl;
		out.setf(oldf);
	}


}

#endif // UTIL_HH
