#include "closed_interval.hh"
#include "utility/const_char.hh"

#include <iostream>

template <class T>
std::ostream&
arma::apmath::operator<<(std::ostream& out, const closed_interval<T>& rhs) {
	return out << '[' << rhs._a << ',' << rhs._b << ']';
}

template <class T>
std::istream&
arma::apmath::operator>>(std::istream& in, closed_interval<T>& rhs) {
	using arma::util::const_char;
	if (!(in >> std::ws >> const_char<'['>())) {
		std::cerr << "Expecting \"[\"." << std::endl;
	}
	in >> rhs._a;
	if (!(in >> std::ws >> const_char<','>())) {
		std::cerr << "Expecting \",\"." << std::endl;
	}
	in >> rhs._b;
	if (!(in >> std::ws >> const_char<']'>())) {
		std::cerr << "Expecting \"]\"." << std::endl;
	}
	return in;
}

template std::ostream&
arma::apmath::operator<<(std::ostream& out, const closed_interval<ARMA_REAL_TYPE>& rhs);

template std::istream&
arma::apmath::operator>>(std::istream& in, closed_interval<ARMA_REAL_TYPE>& rhs);
