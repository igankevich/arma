#include "high_amplitude_solver.hh"
#include "derivative.hh"

#include <complex>

template <class T>
void
arma::velocity::High_amplitude_solver<T>::precompute(
	const Array3D<T>& zeta,
	const int idx_t
) {
	using blitz::Range;
	using blitz::sqrt;
	using blitz::pow;
	Array2D<T> zeta_x(derivative<1,T>(zeta, idx_t));
	Array2D<T> zeta_y(derivative<2,T>(zeta, idx_t));
	Array2D<T> sqrt_zeta(sqrt(T(1) + pow(zeta_x, 2) + pow(zeta_y, 2)));
	this->_zeta_t(idx_t, Range::all(), Range::all()) =
		- derivative<0,T>(zeta, idx_t)
		/ (std::complex<T>(0,1)*((zeta_x + zeta_y)/sqrt_zeta - zeta_x - zeta_y) - T(1));
}

template class arma::velocity::High_amplitude_solver<ARMA_REAL_TYPE>;
