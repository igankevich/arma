#include "apmath/hermite.hh"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

typedef ARMA_REAL_TYPE T;
typedef arma::apmath::Polynomial<ARMA_REAL_TYPE> poly_type;
using arma::apmath::Polynomial;


template <class T>
bool
compare(const Polynomial<T>& lhs, const Polynomial<T>& rhs) noexcept {
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


TEST(HermitePolynomial, Test) {
	using arma::apmath::hermite_polynomial;
	EXPECT_TRUE(compare(hermite_polynomial<T>(0), poly_type{{1}}));
	EXPECT_TRUE(compare(hermite_polynomial<T>(1), poly_type{{0, 1}}));
	EXPECT_TRUE(compare(hermite_polynomial<T>(2), poly_type{{-1, 0, 1, 0}}));
}
