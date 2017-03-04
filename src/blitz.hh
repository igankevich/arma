#ifndef BLITZ_HH
#define BLITZ_HH

#include <cmath>
#include <algorithm>
#include <blitz/array.h>

namespace blitz {

	bool
	isfinite(float rhs) noexcept {
		return std::isfinite(rhs);
	}

	bool
	isfinite(double rhs) noexcept {
		return std::isfinite(rhs);
	}

	BZ_DECLARE_FUNCTION(isfinite);

	int
	div_ceil(int lhs, int rhs) noexcept {
		return lhs/rhs + (lhs%rhs == 0 ? 0 : 1);
	}

	BZ_DECLARE_FUNCTION2(div_ceil);

	template<int n>
	std::ostream&
	operator<<(std::ostream& out, const RectDomain<n>& rhs) {
		return out << rhs.lbound() << " : " << rhs.ubound();
	}

	template <class T>
	T
	length(const TinyVector<T,2>& rhs) noexcept {
		const T u = rhs(0);
		const T v = rhs(1);
		return std::sqrt(u*u + v*v);
	}

	template<int n>
	TinyVector<int, n>
	get_shape(const RectDomain<n>& rhs) {
		return rhs.ubound() - rhs.lbound() + 1;
	}

	template <class T>
	void
	rotate(Array<T,2>& rhs, TinyVector<int,2> point) {
		const int nrows = rhs.rows();
		const int ncols = rhs.cols();
		const int middle_row = point(0);
		const int middle_col = point(1);
		for (int i=0; i<nrows; ++i) {
			for (int j=0; j<middle_col; ++j) {
				std::swap(rhs(i,j), rhs(i,(j+middle_col)%ncols));
			}
		}
		for (int i=0; i<middle_row; ++i) {
			for (int j=0; j<ncols; ++j) {
				std::swap(rhs(i,j), rhs((i+middle_row)%nrows,j));
			}
		}
	}
}

#endif // BLITZ_HH
