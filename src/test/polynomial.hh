#ifndef TEST_POLYNOMIAL_HH
#define TEST_POLYNOMIAL_HH

#include "apmath/polynomial.hh"
#include <cmath>
#include <iostream>

template <class T>
bool
compare(
	const arma::apmath::Polynomial<T>& lhs,
	const arma::apmath::Polynomial<T>& rhs
) noexcept {
	bool result = lhs.order() == rhs.order()
		&& std::equal(
			lhs.data(),
			lhs.data() + lhs.size(),
			rhs.data(),
			[] (T a, T b) { return std::abs(a-b) < T(1e-5); }
		);
	if (!result) {
		std::clog << "lhs=" << lhs << std::endl;
		std::clog << "rhs=" << rhs << std::endl;
	}
	return result;
}

#endif // TEST_POLYNOMIAL_HH
