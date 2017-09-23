#include "wave.hh"

template<class X>
std::ostream&
arma::operator<<(std::ostream& out, const Wave<X>& rhs) {
	return out
		<< rhs._i << ' '
		<< rhs._j << ' '
		<< rhs._wavenum << ' '
		<< rhs._height << ' '
		<< rhs._period;
}

template class arma::Wave<ARMA_REAL_TYPE>;
template std::ostream&
arma::operator<<(std::ostream& out, const Wave<ARMA_REAL_TYPE>& rhs);
