#include "summary.hh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <ostream>

namespace {

	const int colw = 13;

}

template <class T>
void
arma::stats::Summary<T>
::write_quantile_graph() {
	std::string filename;
	std::transform(
		_name.begin(),
		_name.end(),
		std::back_inserter(filename),
		[](char ch) { return !std::isalnum(ch) ? '-' : ch; });
	std::ofstream out(filename);
	out << _graph;
}

template <class T>
std::ostream&
arma::stats::operator<<(std::ostream& out, const Summary<T>& rhs) {
	out.precision(5);
	out << std::setw(colw + 2) << rhs._name;
	out << std::setw(colw) << rhs._mean;
	out << std::setw(colw);
	if (rhs._needvariance) {
		out << rhs._variance;
	} else {
		out << '-';
	}
	out << std::setw(colw) << rhs._expected_mean;
	out << std::setw(colw);
	if (rhs._needvariance) {
		out << rhs._expected_variance;
	} else {
		out << '-';
	}
	out << std::setw(colw) << rhs.qdistance();
	return out;
}

template <class T>
void
arma::stats::Summary<T>
::print_header(std::ostream& out) {
	out << std::setw(colw + 2) << "Property" << std::setw(colw)
	    << "Mean" << std::setw(colw) << "Var" << std::setw(colw)
	    << "ModelMean" << std::setw(colw) << "ModelVar"
	    << std::setw(colw) << "QDistance";
}

template class arma::stats::Summary<ARMA_REAL_TYPE>;

template std::ostream&
arma::stats::operator<<(std::ostream& out, const Summary<ARMA_REAL_TYPE>& rhs);
