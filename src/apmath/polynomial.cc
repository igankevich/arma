#include "polynomial.hh"
#include <iomanip>
#include <algorithm>

template <class T>
arma::apmath::Polynomial<T>::Polynomial(std::initializer_list<T> coefs):
a(coefs.size())
{ std::copy(coefs.begin(), coefs.end(), a.data()); }

template <class T>
arma::apmath::Polynomial<T>&
arma::apmath::Polynomial<T>::operator=(const Polynomial<T>& rhs) {
	const int n = rhs.a.extent(0);
	if (a.extent(0) != n) {
		a.resize(n);
	}
	a = rhs.a;
	return *this;
}


template <class T>
arma::apmath::Polynomial<T>
arma::apmath::Polynomial<T>::operator*(const Polynomial<T>& rhs) const {
	Polynomial<T> tmp(a.size() + rhs.a.size() - 1);
	const int n = a.size();
	const int m = rhs.a.size();
	for (int i=0; i<n; ++i) {
		for (int j=0; j<m; ++j) {
			tmp.a(i+j) += a(i)*rhs.a(j);
		}
	}
	return tmp;
}

template <class T>
std::ostream&
arma::apmath::operator<<(std::ostream& out, const Polynomial<T>& rhs) {
	for (int i=rhs.order()-1; i>1; --i) {
		out << std::setw(16) << std::showpos << std::right
			<< rhs(i) << "x^" << std::noshowpos << i;
	}
	if (rhs.order() > 0) {
		out << std::setw(16) << std::showpos << std::right
			<< rhs(1) << 'x';
	}
	if (rhs.order() >= 0) {
		out << std::setw(18) << std::showpos << std::right
			<< rhs(0);
	}
	return out;
}


template class arma::apmath::Polynomial<ARMA_REAL_TYPE>;
template std::ostream&
arma::apmath::operator<<(std::ostream& out, const Polynomial<ARMA_REAL_TYPE>& rhs);
