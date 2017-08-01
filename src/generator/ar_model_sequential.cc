#include <random>
#include "util.hh"

template <class T>
arma::Array3D<T>
arma::generator::AR_model<T>::do_generate() {
	const T var_wn = this->white_noise_variance();
	write_key_value(std::clog, "White noise variance", var_wn);
	std::mt19937 prng(this->newseed());
	Array3D<T> zeta = generate_white_noise(
		this->_outgrid.size(),
		var_wn,
		std::ref(prng)
	);
	this->generate_surface(zeta, zeta.domain());
	return zeta;
}


