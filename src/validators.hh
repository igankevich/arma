#ifndef VALIDATORS_HH
#define VALIDATORS_HH

#include <stdexcept>
#include "blitz.hh"

namespace arma {

	template<class T, int n>
	void
	validate_shape(const blitz::TinyVector<T,n>& rhs, const char* name) {
		if (blitz::any(rhs <= T(0))) {
			std::cerr
				<< "Bad \"" << name << "\": "
				<< rhs
				<< std::endl;
			throw std::runtime_error("bad parameter");
		}
	}

	template<class T>
	void
	validate_positive(T rhs, const char* name) {
		if (!(rhs > T(0))) {
			std::cerr
				<< "Bad \"" << name << "\": "
				<< rhs
				<< std::endl;
			throw std::runtime_error("bad parameter");
		}
	}

	template<class T, int n>
	void
	validate_finite(const blitz::TinyVector<T,n>& rhs, const char* name) {
		using blitz::all;
		using blitz::isfinite;
		if (!all(isfinite(rhs))) {
			std::cerr
				<< "Bad \"" << name << "\": "
				<< rhs
				<< std::endl;
			throw std::runtime_error("bad parameter");
		}
	}

	template<class T>
	void
	validate_finite(T rhs, const char* name) {
		if (!std::isfinite(rhs)) {
			std::cerr
				<< "Bad \"" << name << "\": "
				<< rhs
				<< std::endl;
			throw std::runtime_error("bad parameter");
		}
	}

}

#endif // VALIDATORS_HH
