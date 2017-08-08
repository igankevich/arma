#include "profile.hh"

template <class T>
arma::Array3D<T>
arma::generator::AR_model<T>::do_generate() {
	ARMA_PROFILE_START(generate_white_noise);
	Array3D<T> zeta = this->generate_white_noise();
	ARMA_PROFILE_END(generate_white_noise);
	this->generate_surface(zeta, zeta.domain());
	return zeta;
}


