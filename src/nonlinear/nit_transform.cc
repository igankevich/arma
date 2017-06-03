#include "nit_transform.hh"
#include "params.hh"
#include "validators.hh"
#include "domain.hh"
#include "transforms.hh"

#include <string>
#include <stdexcept>
#include <iostream>
#include <cmath>

template <class T>
void
arma::nonlinear::NIT_transform<T>::transform_CDF(Array3D<T> acf) {
	// Do nothing, if target distribution matches the current one.
	if (_targetdist == bits::Distribution::Normal) return;
	const T stdev = std::sqrt(acf(0,0,0));
	const T breadth = _nsigma*stdev;
	_cdfsolver.interval(-breadth, breadth);
	const Domain<T,1> grid(
		Vec1D<T>(-breadth),
		Vec1D<T>(breadth),
		Vec1D<int>(_intnodes)
	);
	auto nodes = transform_CDF(
		grid,
		normaldist_type(T(0), stdev),
		_skewnormal,
		_cdfsolver
	);
	_xnodes.reference(nodes.first);
	_ynodes.reference(nodes.second);
}

template <class T>
void
arma::nonlinear::NIT_transform<T>::interpolate_CDF() {
	_intcoefs = linalg::interpolate(_xnodes, _ynodes, _intcoefs.size());
}

template <class T>
void
arma::nonlinear::NIT_transform<T>::expand_into_gram_charlier_series(
	Array3D<T> acf
) {
	gram_charlier_expand(
		_intcoefs,
		_gcscoefs.numElements(),
		acf(0,0,0)
	);
}

template <class T>
void
arma::nonlinear::NIT_transform<T>::do_transform_ACF(Array3D<T>& acf) {
	_acfsolver.interval(_acfinterval, _acfinterval);
	transform_ACF(
		acf.data(),
		acf.numElements(),
		_gcscoefs,
		_acfsolver
	);
}

template <class T>
void
arma::nonlinear::NIT_transform<T>::transform_ACF(Array3D<T>& acf) {
	transform_CDF(acf);
	interpolate_CDF();
	expand_into_gram_charlier_series(acf);
	do_transform_ACF(acf);
}

template <class T>
void
arma::nonlinear::NIT_transform<T>::transform_realisation(
	Array3D<T> acf,
	Array3D<T>& realisation
) {
	const T stdev = std::sqrt(acf(0,0,0));
	transform_data(
		realisation.data(),
		realisation.numElements(),
		normaldist_type(T(0), stdev),
		_skewnormal,
		_cdfsolver
	);
}

std::istream&
arma::nonlinear::bits::operator>>(std::istream& in, Distribution& rhs) {
	std::string name;
	in >> std::ws >> name;
	if (name == "normal") {
		rhs = Distribution::Normal;
	} else if (name == "skew_normal") {
		rhs = Distribution::Skew_normal;
	} else {
		in.setstate(std::ios::failbit);
		std::clog << "Invalid distribution: " << name << std::endl;
		throw std::runtime_error("bad distribution");
	}
	return in;
}

const char*
arma::nonlinear::bits::to_string(Distribution rhs) {
	switch (rhs) {
		case Distribution::Normal: return "normal";
		case Distribution::Skew_normal: return "skew_normal";
		default: return "UNKNOWN";
	}
}

std::ostream&
arma::nonlinear::bits::operator<<(std::ostream& out, const Distribution& rhs) {
	return out << to_string(rhs);
}

template <class T>
std::ostream&
arma::nonlinear::operator<<(std::ostream& out, const NIT_transform<T>& rhs) {
	out << "dist=" << rhs._targetdist << ',';
	switch (rhs._targetdist) {
		case bits::Distribution::Normal:
			out << rhs._normal;
			break;
		case bits::Distribution::Skew_normal:
			out << rhs._skewnormal;
			break;
		default:
			break;
	}
	out << ",interpolation_nodes=" << rhs._xnodes.numElements()
		<< ",interpolation_order=" << rhs._intnodes
		<< ",gram_charlier_order=" << rhs._gcscoefs.numElements()
		;
	return out;
}

template <class T>
std::istream&
arma::nonlinear::operator>>(std::istream& in, NIT_transform<T>& rhs) {
	int intorder = NIT_transform<T>::default_interpolation_order;
	int gcsorder = NIT_transform<T>::default_gram_charlier_order;
	sys::parameter_map params({
	    {"distribution", sys::make_param(rhs._targetdist)},
	    {"interpolation_order", sys::make_param(intorder, validate_positive<T>)},
	    {"interpolation_nodes", sys::make_param(rhs._intnodes, validate_positive<T>)},
	    {"gram_charlier_order", sys::make_param(gcsorder, validate_positive<T>)},
	    {"nsigma", sys::make_param(rhs._nsigma, validate_positive<T>)},
	    {"acf_interval", sys::make_param(rhs._acfinterval, validate_positive<T>)},
	    {"cdf_solver", sys::make_param(rhs._cdfsolver)},
	    {"acf_solver", sys::make_param(rhs._acfsolver)},
	}, "nit_transform", true);
	in >> params;
	rhs._intcoefs.resize(intorder);
	rhs._gcscoefs.resize(gcsorder);
	return in;
}

template <class T>
std::istream&
arma::nonlinear::operator>>(
	std::istream& in,
	typename NIT_transform<T>::Distribution_wrapper& rhs
) {
	switch (rhs._obj._targetdist) {
		case bits::Distribution::Normal:
			in >> rhs._obj._normal;
			break;
		case bits::Distribution::Skew_normal:
			in >> rhs._obj._skewnormal;
			break;
		default:
			break;
	}
	return in;
}

template std::ostream&
arma::nonlinear::operator<<(
	std::ostream& out,
	const NIT_transform<ARMA_REAL_TYPE>& rhs
);

template std::istream&
arma::nonlinear::operator>>(
	std::istream& in,
	NIT_transform<ARMA_REAL_TYPE>& rhs
);
