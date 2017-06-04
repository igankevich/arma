#include "skew_normal.hh"

template <class T>
std::istream&
arma::stats::operator>>(std::istream& in, Skew_normal<T>& rhs) {
	sys::parameter_map params({
	    {"mean", sys::make_param(rhs._mean)},
	    {"stdev", sys::make_param(rhs._sigma)},
	    {"alpha", sys::make_param(rhs._alpha)},
	}, true);
	in >> params;
	return in;
}

template <class T>
std::ostream&
arma::stats::operator<<(std::ostream& out, const Skew_normal<T>& rhs) {
	return out
		<< "mean=" << rhs._mean
		<< ",stdev=" << rhs._sigma
		<< ",alpha=" << rhs._alpha;
}

template std::istream&
arma::stats::operator>>(std::istream& in, Skew_normal<ARMA_REAL_TYPE>& rhs);

template std::ostream&
arma::stats::operator<<(std::ostream& out, const Skew_normal<ARMA_REAL_TYPE>& rhs);
