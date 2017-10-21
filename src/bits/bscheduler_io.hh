#ifndef BITS_BSCHEDULER_IO_HH
#define BITS_BSCHEDULER_IO_HH

#include <vector>

#include "domain.hh"
#include "grid.hh"
#include "types.hh"

template <class T, int N>
sys::pstream&
operator<<(sys::pstream& out, const arma::Domain<T,N>& rhs) {
	return out << rhs._lbound << rhs._ubound << rhs._npoints;
}

template <class T, int N>
sys::pstream&
operator>>(sys::pstream& in, arma::Domain<T,N>& rhs) {
	return in >> rhs._lbound >> rhs._ubound >> rhs._npoints;
}

namespace std {

	template <class T>
	sys::pstream&
	operator<<(sys::pstream& out, const std::vector<T>& rhs) {
		uint32_t n = rhs.size();
		out << n;
		for (uint32_t i=0; i<n; ++i) {
			out << rhs[i];
		}
		return out;
	}

	template <class T>
	sys::pstream&
	operator>>(sys::pstream& in, std::vector<T>& rhs) {
		uint32_t n = 0;
		in >> n;
		rhs.resize(n);
		for (uint32_t i=0; i<n; ++i) {
			in >> rhs[i];
		}
		return in;
	}

}

#endif // vim:filetype=cpp
