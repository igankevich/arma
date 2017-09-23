#include "small_amplitude_solver.hh"
#include "derivative.hh"
#include "physical_constants.hh"
#include "small_amplitude_solver/integrals.hh"

template <class T>
arma::Array2D<T>
arma::velocity::Small_amplitude_solver<T>::compute_velocity_field_2d(
	const Discrete_function<T,3>& zeta,
	const Shape2D arr_size,
	const T z,
	const int idx_t
) {
	Array2D<T> phi(arr_size);
	return phi;
}

template class arma::velocity::Small_amplitude_solver<ARMA_REAL_TYPE>;
