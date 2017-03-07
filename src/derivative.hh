#ifndef DERIVATIVE_HH
#define DERIVATIVE_HH

#include "types.hh"

namespace arma {

	template <int dimension, class T, class O=T>
	Array2D<O>
	derivative(const Array3D<T>& rhs, const int idx_t) {
		assert(dimension >= 0);
		assert(dimension < 3);
		const int min_d = 0;
		const int max_d = rhs.extent(dimension)-1;
		const int nx = rhs.extent(1);
		const int ny = rhs.extent(2);
		Array2D<O> result(size2(nx, ny));
		if (max_d - min_d < 2) {
			throw std::length_error("bad shape");
		}
		for (int i=0; i<nx; ++i) {
			for (int j=0; j<ny; ++j) {
				size3 idx(idx_t, i, j);
				const int d = idx(dimension);
				if (d == min_d) {
					/**
					1. Compute forward differences on the left border.
					\f[
						f'_i = \frac{-f_{i+2} + 4f_{i+1} - 3f_{i}}{2}
					\f]
					*/
					size3 idx1(idx);
					++idx1(dimension);
					size3 idx2(idx1);
					++idx2(dimension);
					result(i,j) = T(0.5)*(-rhs(idx2) + T(4)*rhs(idx1) - T(3)*rhs(idx));
				} else if (d == max_d) {
					/**
					2. Compute backward differences on the right border.
					\f[
						f'_i = \frac{3f_{i} - 4f_{i-1} + f_{i-2}}{2}
					\f]
					*/
					size3 idx1(idx);
					--idx1(dimension);
					size3 idx2(idx1);
					--idx2(dimension);
					result(i,j) = T(0.5)*(rhs(idx) - T(4)*rhs(idx1) + rhs(idx2));
				} else {
					/**
					3. Compute central differences in all other points.
					\f[
						f'_i = \frac{f_{i+1} - f_{i-1}}{2}
					\f]
					*/
					size3 idx0(idx);
					--idx0(dimension);
					size3 idx1(idx);
					++idx1(dimension);
					result(i,j) = T(0.5)*(rhs(idx1) - rhs(idx0));
				}
			}
		}
		return result;
	}

}

#endif // DERIVATIVE_HH
