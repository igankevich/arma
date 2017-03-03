#include <blitz/array.h>

#ifndef BLITZ_HH
#define BLITZ_HH

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
}

#endif // BLITZ_HH
