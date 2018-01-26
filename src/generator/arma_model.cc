#include "arma_model.hh"

#include "bits/acf_wrapper.hh"
#include "params.hh"
#include "validators.hh"

template <class T>
void
arma::generator::ARMA_model<T>
::validate() const {
	AR_model<T>::validate();
	std::clog << "AR process is OK." << std::endl;
	MA_model<T>::validate();
}

template <class T>
void
arma::generator::ARMA_model<T>
::determine_coefficients() {
	using namespace blitz;
	if (product(AR_model<T>::order()) > 0) {
		AR_model<T>::determine_coefficients();
	}
//	ma_model::recompute_acf(_acf_orig, ar_model::coefficients());
	MA_model<T>::determine_coefficients();
	// TODO: set white noise variance
}

template <class T>
T
arma::generator::ARMA_model<T>
::white_noise_variance(Array3D<T> phi, Array3D<T> theta) const {
	return AR_model<T>::white_noise_variance(phi) *
	       MA_model<T>::white_noise_variance(theta) /
	       AR_model<T>::acf_variance();
}

template <class T>
void
arma::generator::ARMA_model<T>
::generate_surface(
	Array3D<T>& zeta,
	Array3D<T>& eps,
	const Domain3D& subdomain
) {
	MA_model<T>::generate_surface(zeta, eps, subdomain);
	AR_model<T>::generate_surface(zeta, subdomain);
}

template <class T>
void
arma::generator::ARMA_model<T>
::read(std::istream& in) {
	typedef typename Basic_model<T>::grid_type grid_type;
	using blitz::shape;
	bits::ACF_wrapper<T> acf_wrapper(this->_acf_orig);
	sys::parameter_map params {
		{
			{"ar_model", sys::make_param(static_cast<AR_model<T>&>(*this))},
			{"ma_model", sys::make_param(static_cast<MA_model<T>&>(*this))},
		},
		true
	};
	params.insert({{"acf", sys::make_param(acf_wrapper)}});
	params.insert(this->AR_model<T>::parameters());
	in >> params;
	validate_shape(this->_acf_orig.shape(), "arma_model.acf_orig.shape");
	this->AR_model<T>::setacf(
		ARMA_model<T>::slice_front(
			this->_acf_orig,
			this->AR_model<T>::order()
		)
	);
	this->AR_model<T>::setgrid(this->ARMA_model<T>::grid());
	this->MA_model<T>::setacf(
		ARMA_model<T>::slice_front(
			this->_acf_orig,
			this->MA_model<T>::order()
		)
	);
	this->MA_model<T>::setgrid(this->ARMA_model<T>::grid());
	// resize output grid to match ACF delta size
	this->_outgrid =
		grid_type(
			this->_outgrid.num_points(),
			this->_acf_orig.grid().delta() * this->_outgrid.num_patches() * T(1.0)
		);
	std::clog << "ar_model" << static_cast<const AR_model<T>&>(*this) <<
	    std::endl;
	std::clog << "ma_model" << static_cast<const MA_model<T>&>(*this) <<
	    std::endl;
}

template <class T>
void
arma::generator::ARMA_model<T>
::write(std::ostream& out) const {
	out << "ar_model=";
	this->AR_model<T>::write(out);
	out << ",ma_model=";
	this->MA_model<T>::write(out);
}

template class arma::generator::ARMA_model<ARMA_REAL_TYPE>;
