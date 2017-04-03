#include "high_amplitude_realtime_solver.hh"

template <class T>
arma::Array4D<T>
arma::velocity::High_amplitude_realtime_solver<T>::operator()(
	const Array3D<T>& zeta
) {
	return Array4D<T>();
}

template class arma::velocity::High_amplitude_realtime_solver<ARMA_REAL_TYPE>;
