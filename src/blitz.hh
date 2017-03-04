#ifndef BLITZ_HH
#define BLITZ_HH

#include <cmath>
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
}

#endif // BLITZ_HH
