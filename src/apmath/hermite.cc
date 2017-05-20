#include "hermite.hh"
#include <algorithm>

template <class T>
arma::apmath::Polynomial<T>
arma::apmath::hermite_polynomial(int n) {
	typedef blitz::Array<T,1> array_type;
	if (n == 0) {
		Polynomial<T> p(0);
		p(0) = 1;  // 1
		return p;
	}
	if (n == 1) {
		Polynomial<T> p(1);
		p(0) = 0;  // x
		p(1) = 1;
		return p;
	}
	const int m = n+1;
	// in an order of decreasing x powers
	array_type h0(m), h1(m);
	h0(0) = T(1);
	h1(0) = T(1);
	h1(1) = T(0);
	Polynomial<T> hn(n+1);
	int h1_size = 2;
	int h0_size = 1;
	for (int i=2; i<m; i++) {
		const int hn_size = h1_size+1;
		for (int j=0; j<h1_size; j++) {
			hn(j) = h1(j);
		}
		hn(h1_size) = T(0);
		for (int j=0; j<h0_size; j++) {
			hn(hn_size-h0_size+j) -= (i-1)*h0(j);
		}
		for (int j=0; j<h1_size; j++) {
			h0(j) = h1(j);
		}
		h0_size = h1_size;
		for (int j=0; j<h1_size+1; j++) {
			h1(j) = hn(j);
		}
		++h1_size;
	}
	std::reverse(&hn[0], &hn[hn.order()]);
	return hn;
}


template arma::apmath::Polynomial<ARMA_REAL_TYPE>
arma::apmath::hermite_polynomial(int n);
