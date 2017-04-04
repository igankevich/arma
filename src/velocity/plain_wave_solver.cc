#include "plain_wave_solver.hh"
#include "physical_constants.hh"
#include "validators.hh"
#include "params.hh"

template <class T>
arma::Array2D<T>
arma::velocity::Plain_wave_solver<T>::compute_velocity_field_2d(
	const Discrete_function<T,3>& zeta,
	const Shape2D arr_size,
	const T z,
	const int idx_t
) {
	using constants::_2pi;
	typedef typename wave_type::array_type array_type;
	const array_type& A = _waves.amplitudes();
	const array_type& omega = _waves.velocities();
	const array_type& k = _waves.wavenumbers();
	const array_type& phases = _waves.phases();
	const T shift = _waves.get_shift();
	Array2D<T> phi(arr_size);
	const int nx = arr_size(0);
	const int ny = arr_size(1);
	const T h = this->_depth;
	for (int i=0; i<nx; ++i) {
		for (int j=0; j<ny; ++j) {
			phi(i,j) = blitz::sum(
				T(2)*A*omega
				*blitz::cos(_2pi<T>*k*i - omega*idx_t + shift + phases)
				*blitz::sinh(_2pi<T>*k*(z + h))
				/k
				/blitz::sinh(_2pi<T>*k*h)
			);
		}
	}
	return phi;
}

template <class T>
void
arma::velocity::Plain_wave_solver<T>::write(std::ostream& out) const {
	out << "waves=" << this->_waves << ','
		<< "depth=" << this->_depth << ','
		<< "domain=" << this->_domain;
}

template <class T>
void
arma::velocity::Plain_wave_solver<T>::read(std::istream& in) {
	sys::parameter_map params({
		{"waves", sys::make_param(this->_waves)},
		{"depth", sys::make_param(this->_depth, validate_finite<T>)},
		{"domain", sys::make_param(this->_domain, validate_domain<T,2>)},
	}, true);
	in >> params;
}

template class arma::velocity::Plain_wave_solver<ARMA_REAL_TYPE>;
