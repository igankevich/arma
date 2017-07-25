#include "polynomial.hh"
#include <iomanip>
#include <algorithm>
#include <cmath>

template <class T>
arma::apmath::Polynomial<T>::Polynomial(std::initializer_list<T> coefs):
a(coefs.size())
{ std::copy(coefs.begin(), coefs.end(), a.data()); }

template <class T>
arma::apmath::Polynomial<T>&
arma::apmath::Polynomial<T>::operator=(const Polynomial<T>& rhs) {
	const size_t n = rhs.size();
	if (a.size() != n) {
		a.resize(n);
	}
	a = rhs.a;
	return *this;
}

template <class T>
arma::apmath::Polynomial<T>
arma::apmath::operator*(const Polynomial<T>& lhs, const Polynomial<T>& rhs) {
	const int n = lhs.size();
	const int m = rhs.size();
	Polynomial<T> tmp(n + m - 1);
	for (int i=0; i<n; ++i) {
		for (int j=0; j<m; ++j) {
			tmp(i+j) += lhs(i)*rhs(j);
		}
	}
	return tmp;
}

template <class T>
arma::apmath::Polynomial<T>&
arma::apmath::Polynomial<T>::normalise(T eps) {
	if (size() > 1) {
		int i = order();
		while (i>1 && std::abs(a[i]) < eps) {
			--i;
		}
		a.resize(i);
	}
	return *this;
}

template <class T>
std::ostream&
arma::apmath::operator<<(std::ostream& out, const Polynomial<T>& rhs) {
	if (rhs.size() > 0) {
		for (int i=rhs.size()-1; i>1; --i) {
			out << std::setw(16) << std::showpos << std::right
				<< rhs(i) << "x^" << std::noshowpos << i;
		}
	}
	if (rhs.size() > 1) {
		out << std::setw(16) << std::showpos << std::right
			<< rhs(1) << 'x' << std::noshowpos;
	}
	if (rhs.size() > 0) {
		out << std::setw(18) << std::showpos << std::right
			<< rhs(0) << std::noshowpos;
	}
	return out;
}


template class arma::apmath::Polynomial<ARMA_REAL_TYPE>;
template std::ostream&
arma::apmath::operator<<(std::ostream& out, const Polynomial<ARMA_REAL_TYPE>& rhs);

template arma::apmath::Polynomial<ARMA_REAL_TYPE>
arma::apmath::operator*(
	const Polynomial<ARMA_REAL_TYPE>& lhs,
	const Polynomial<ARMA_REAL_TYPE>& rhs
);
