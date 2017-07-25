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
void
arma::generator::ARMA_model<T>::read(std::istream& in) {
	ACF_wrapper<T> acf_wrapper(this->_acf_orig);
	sys::parameter_map params({
		{"acf", sys::make_param(acf_wrapper)},
		{"ar_model", sys::make_param(static_cast<AR_model<T>&>(*this))},
		{"ma_model", sys::make_param(static_cast<MA_model<T>&>(*this))},
		{"out_grid", sys::make_param(this->_outgrid, validate_grid<T,3>)},
	}, true);
	in >> params;
	validate_shape(this->_acf_orig.shape(), "ma_model.acf.shape");
	this->AR_model<T>::setacf(ARMA_model<T>::slice_front(
		this->_acf_orig,
		this->AR_model<T>::order()
	));
	this->AR_model<T>::setgrid(this->ARMA_model<T>::grid());
	this->MA_model<T>::setacf(ARMA_model<T>::slice_back(
		this->_acf_orig,
		this->MA_model<T>::order()
	));
	this->MA_model<T>::setgrid(this->ARMA_model<T>::grid());
}

template <class T>
void
arma::generator::ARMA_model<T>::write(std::ostream& out) const {
	out << "ar_model=";
	this->AR_model<T>::write(out);
	out << ",ma_model=";
	this->MA_model<T>::write(out);
}

template class arma::generator::ARMA_model<ARMA_REAL_TYPE>;
