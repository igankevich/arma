#include "wave.hh"

template<class X>
std::ostream&
arma::operator<<(std::ostream& out, const Wave<X>& rhs) {
	return out
		<< rhs.i << ' '
		<< rhs.j << ' '
		<< rhs.k << ' '
		<< rhs.height << ' '
		<< rhs.period;
}

template class arma::Wave<ARMA_REAL_TYPE>;
template std::ostream&
arma::operator<<(std::ostream& out, const Wave<ARMA_REAL_TYPE>& rhs);
