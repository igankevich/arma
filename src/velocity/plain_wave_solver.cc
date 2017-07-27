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
	using std::cos;
	using std::sinh;
	using std::sqrt;
	const T shift = this->_waves.get_shift();
	Array2D<T> phi(arr_size);
	const int nx = arr_size(0);
	const int ny = arr_size(1);
	const T h = this->_depth;
	const Domain<T,3> dom(
		zeta.grid().length(),
		zeta.shape()
	);
	const int nwaves = this->_waves.num_waves();
	for (int i=0; i<nx; ++i) {
		for (int j=0; j<ny; ++j) {
			const T x = dom(i, 1);
			const T y = dom(j, 2);
			T sum = 0;
			for (int k=0; k<nwaves; ++k) {
				const T a = this->_waves.amplitude(k);
				const T kx = this->_waves.wavenum_x(k);
				const T ky = this->_waves.wavenum_y(k);
				const T klen = sqrt(kx*kx + ky*ky);
				const T klen2pi = _2pi<T>*klen;
				const T w = this->_waves.velocity(k);
				const T p = this->_waves.phase(k);
				sum += T(2)*a*w
					* cos(_2pi<T>*(kx*x + ky*y) - w*idx_t + shift + p)
					* sinh(klen2pi*(z + h))
					/ klen
					/ sinh(klen2pi*h);
			}
			phi(i,j) = sum;
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
