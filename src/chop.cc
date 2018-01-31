#include "chop.hh"


template <class T>
arma::Shape3D
arma::chop_right(const Array3D<T>& rhs, T eps) {
	using blitz::abs;
	using blitz::all;
	using blitz::Range;
	// shrink third dimension
	int k = rhs.extent(2) - 1;
	while (k >= 1 && all(abs(rhs(Range::all(), Range::all(), k)) < eps)) {
		--k;
	}
	// shrink second dimension
	int j = rhs.extent(1) - 1;
	while (j >= 1 && all(abs(rhs(Range::all(), j, Range(0,k))) < eps)) {
		--j;
	}
	// shrink first dimension
	int i = rhs.extent(0) - 1;
	while (i >= 1 && all(abs(rhs(i, Range(0,j), Range(0,k))) < eps)) {
		--i;
	}
	return Shape3D(i+1,j+1,k+1);
}


template arma::Shape3D
arma::chop_right(const Array3D<ARMA_REAL_TYPE>& rhs, ARMA_REAL_TYPE eps);
