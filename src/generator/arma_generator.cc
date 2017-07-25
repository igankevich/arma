#include "arma_generator.hh"

#include "validators.hh"
#include "params.hh"
#include "acf.hh"

template <class T>
T
arma::generator::ARMA_generator<T>::white_noise_variance() const {
	return white_noise_variance(AR_generator<T>::coefficients(),
								MA_generator<T>::coefficients());
}

template <class T>
void
arma::generator::ARMA_generator<T>::validate() const {
	AR_generator<T>::validate();
	std::clog << "AR process is OK." << std::endl;
	MA_generator<T>::validate();
}

template <class T>
void
arma::generator::ARMA_generator<T>::determine_coefficients() {
	using namespace blitz;
	if (product(AR_generator<T>::order()) > 0) {
		AR_generator<T>::determine_coefficients();
	}
//	ma_model::recompute_acf(_acf_orig, ar_model::coefficients());
	MA_generator<T>::determine_coefficients();
}

template <class T>
T
arma::generator::ARMA_generator<T>::white_noise_variance(Array3D<T> phi, Array3D<T> theta) const {
	return AR_generator<T>::white_noise_variance(phi) *
		   MA_generator<T>::white_noise_variance(theta) /
		   AR_generator<T>::acf_variance();
}

template <class T>
void
arma::generator::ARMA_generator<T>::operator()(
	Array3D<T>& zeta,
	Array3D<T>& eps,
	const Domain3D& subdomain
) {
	MA_generator<T>::operator()(zeta, eps, subdomain);
	AR_generator<T>::operator()(zeta, zeta, subdomain);
}

template <class T>
std::istream&
arma::generator::operator>>(std::istream& in, ARMA_generator<T>& rhs) {
	ACF_wrapper<T> acf_wrapper(rhs._acf_orig);
	sys::parameter_map params({
		{"acf", sys::make_param(acf_wrapper)},
		{"ar_model", sys::make_param(static_cast<AR_generator<T>&>(rhs))},
		{"ma_model", sys::make_param(static_cast<MA_generator<T>&>(rhs))},
	}, true);
	in >> params;
	validate_shape(rhs._acf_orig.shape(), "ma_model.acf.shape");
	rhs.AR_generator<T>::setacf(ARMA_generator<T>::slice_front(
		rhs._acf_orig,
		rhs.AR_generator<T>::order()
	));
	rhs.MA_generator<T>::setacf(ARMA_generator<T>::slice_back(
		rhs._acf_orig,
		rhs.MA_generator<T>::order()
	));
	return in;
}

template class arma::generator::ARMA_generator<ARMA_REAL_TYPE>;

template std::istream&
arma::generator::operator>>(std::istream& in, ARMA_generator<ARMA_REAL_TYPE>& rhs);
