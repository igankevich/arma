#include "distribution.hh"
#include "params.hh"

template <class T>
std::istream&
arma::stats::operator>>(std::istream& in, Gaussian<T>& rhs) {
	sys::parameter_map params({
	    {"mean", sys::make_param(rhs._mean)},
	    {"stdev", sys::make_param(rhs._sigma)},
	}, true);
	in >> params;
	return in;
}


template <class T>
std::ostream&
arma::stats::operator<<(std::ostream& out, const Gaussian<T>& rhs) {
	return out
		<< "mean=" << rhs._mean
		<< ",stdev=" << rhs._sigma;
}

template <class T>
std::istream&
arma::stats::operator>>(std::istream& in, Gram_Charlier<T>& rhs) {
	sys::parameter_map params({
	    {"skewness", sys::make_param(rhs._skewness)},
	    {"kurtosis", sys::make_param(rhs._kurtosis)},
	}, true);
	in >> params;
	return in;
}


template <class T>
std::ostream&
arma::stats::operator<<(std::ostream& out, const Gram_Charlier<T>& rhs) {
	return out
		<< "skewness=" << rhs._skewness
		<< ",kurtosis=" << rhs._kurtosis;
}

template std::istream&
arma::stats::operator>>(std::istream& in, Gaussian<ARMA_REAL_TYPE>& rhs);

template std::ostream&
arma::stats::operator<<(std::ostream& out, const Gaussian<ARMA_REAL_TYPE>& rhs);

template std::istream&
arma::stats::operator>>(std::istream& in, Gram_Charlier<ARMA_REAL_TYPE>& rhs);

template std::ostream&
arma::stats::operator<<(std::ostream& out, const Gram_Charlier<ARMA_REAL_TYPE>& rhs);
