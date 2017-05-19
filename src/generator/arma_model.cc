#include "arma_model.hh"

#include "validators.hh"
#include "params.hh"
#include "acf.hh"

template <class T>
T
arma::generator::ARMA_model<T>::white_noise_variance() const {
	return white_noise_variance(AR_model<T>::coefficients(),
								MA_model<T>::coefficients());
}

template <class T>
void
arma::generator::ARMA_model<T>::validate() const {
	AR_model<T>::validate();
	std::clog << "AR process is OK." << std::endl;
	MA_model<T>::validate();
}

template <class T>
void
arma::generator::ARMA_model<T>::determine_coefficients() {
	using namespace blitz;
	if (product(AR_model<T>::order()) > 0) {
		AR_model<T>::determine_coefficients();
	}
//	ma_model::recompute_acf(_acf_orig, ar_model::coefficients());
	MA_model<T>::determine_coefficients();
}

template <class T>
T
arma::generator::ARMA_model<T>::white_noise_variance(Array3D<T> phi, Array3D<T> theta) const {
	return AR_model<T>::white_noise_variance(phi) *
		   MA_model<T>::white_noise_variance(theta) /
		   AR_model<T>::acf_variance();
}

template <class T>
void
arma::generator::ARMA_model<T>::operator()(
	Array3D<T>& zeta,
	Array3D<T>& eps,
	const Domain3D& subdomain
) {
	MA_model<T>::operator()(zeta, eps, subdomain);
	AR_model<T>::operator()(zeta, zeta, subdomain);
}

template <class T>
std::istream&
arma::generator::operator>>(std::istream& in, ARMA_model<T>& rhs) {
	ACF_wrapper<T> acf_wrapper(rhs._acf_orig);
	sys::parameter_map params({
		{"acf", sys::make_param(acf_wrapper)},
		{"ar_model", sys::make_param(static_cast<AR_model<T>&>(rhs))},
		{"ma_model", sys::make_param(static_cast<MA_model<T>&>(rhs))},
	}, true);
	in >> params;
	validate_shape(rhs._acf_orig.shape(), "ma_model.acf.shape");
	rhs.AR_model<T>::setacf(ARMA_model<T>::slice_front(
		rhs._acf_orig,
		rhs.AR_model<T>::order()
	));
	rhs.MA_model<T>::setacf(ARMA_model<T>::slice_back(
		rhs._acf_orig,
		rhs.MA_model<T>::order()
	));
	return in;
}

template class arma::generator::ARMA_model<ARMA_REAL_TYPE>;

template std::istream&
arma::generator::operator>>(std::istream& in, ARMA_model<ARMA_REAL_TYPE>& rhs);
