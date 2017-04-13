#include "arma_model.hh"

#include "validators.hh"
#include "params.hh"
#include "acf.hh"

template <class T>
T
arma::ARMA_model<T>::white_noise_variance() const {
	return white_noise_variance(Autoregressive_model<T>::coefficients(),
								Moving_average_model<T>::coefficients());
}

template <class T>
void
arma::ARMA_model<T>::validate() const {
	Autoregressive_model<T>::validate();
	std::clog << "AR process is OK." << std::endl;
	Moving_average_model<T>::validate();
}

template <class T>
void
arma::ARMA_model<T>::determine_coefficients() {
	using namespace blitz;
	if (product(Autoregressive_model<T>::order()) > 0) {
		Autoregressive_model<T>::determine_coefficients();
	}
//	ma_model::recompute_acf(_acf_orig, ar_model::coefficients());
	Moving_average_model<T>::determine_coefficients();
}

template <class T>
T
arma::ARMA_model<T>::white_noise_variance(Array3D<T> phi, Array3D<T> theta) const {
	return Autoregressive_model<T>::white_noise_variance(phi) *
		   Moving_average_model<T>::white_noise_variance(theta) /
		   Autoregressive_model<T>::acf_variance();
}

template <class T>
void
arma::ARMA_model<T>::operator()(
	Array3D<T>& zeta,
	Array3D<T>& eps,
	const Domain3D& subdomain
) {
	Moving_average_model<T>::operator()(zeta, eps, subdomain);
	Autoregressive_model<T>::operator()(zeta, zeta, subdomain);
}

template <class T>
std::istream&
arma::operator>>(std::istream& in, ARMA_model<T>& rhs) {
	ACF_wrapper<T> acf_wrapper(rhs._acf_orig);
	sys::parameter_map params({
		{"acf", sys::make_param(acf_wrapper)},
		{"ar_model", sys::make_param(static_cast<Autoregressive_model<T>&>(rhs))},
		{"ma_model", sys::make_param(static_cast<Moving_average_model<T>&>(rhs))},
	}, true);
	in >> params;
	validate_shape(rhs._acf_orig.shape(), "ma_model.acf.shape");
	rhs.Autoregressive_model<T>::setacf(ARMA_model<T>::slice_front(
		rhs._acf_orig,
		rhs.Autoregressive_model<T>::order()
	));
	rhs.Moving_average_model<T>::setacf(ARMA_model<T>::slice_back(
		rhs._acf_orig,
		rhs.Moving_average_model<T>::order()
	));
	return in;
}

template class arma::ARMA_model<ARMA_REAL_TYPE>;

template std::istream&
arma::operator>>(std::istream& in, ARMA_model<ARMA_REAL_TYPE>& rhs);
