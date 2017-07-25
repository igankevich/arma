#include <random>
#include "util.hh"

template <class T>
arma::Array3D<T>
arma::generator::Basic_ARMA_model<T>::do_generate() {
	const T var_wn = this->white_noise_variance();
	write_key_value(std::clog, "White noise variance", var_wn);
	std::mt19937 prng;
	prng.seed(newseed());
	Array3D<T> eps = generate_white_noise(
		this->_outgrid.size(),
		var_wn,
		std::ref(prng)
	);
	Array3D<T> zeta(eps.shape());
	this->operator()(zeta, eps, zeta.domain());
	return zeta;
}


