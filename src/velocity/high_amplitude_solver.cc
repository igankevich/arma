#include "high_amplitude_solver.hh"
#include "derivative.hh"

#include <complex>

template <class T>
void
arma::velocity::High_amplitude_solver<T>::precompute(
	const Discrete_function<T,3>& zeta,
	const int idx_t
) {
	using blitz::Range;
	using blitz::sqrt;
	using blitz::pow;
	Array2D<T> zeta_x(derivative<1,T>(zeta, zeta.grid().delta(), idx_t));
	Array2D<T> zeta_y(derivative<2,T>(zeta, zeta.grid().delta(), idx_t));
	Array2D<T> sqrt_zeta(sqrt(T(1) + pow(zeta_x, 2) + pow(zeta_y, 2)));
	const std::complex<T> I(0,1);
	this->_zeta_t(idx_t, Range::all(), Range::all()) =
		derivative<0,T>(zeta, zeta.grid().delta(), idx_t)
		/ (I*((zeta_x + zeta_y)/sqrt_zeta - zeta_x - zeta_y) - T(1)/sqrt_zeta);
	this->compute_wave_number_range_from_surface(zeta, idx_t);
}

template class arma::velocity::High_amplitude_solver<ARMA_REAL_TYPE>;
