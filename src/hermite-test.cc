#include "apmath/hermite.hh"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include "test/polynomial.hh"

typedef ARMA_REAL_TYPE T;
typedef arma::apmath::Polynomial<ARMA_REAL_TYPE> poly_type;
using arma::apmath::Polynomial;


TEST(HermitePolynomial, Test) {
	using arma::apmath::hermite_polynomial;
	EXPECT_TRUE(compare(hermite_polynomial<T>(0), poly_type{{1}}));
	EXPECT_TRUE(compare(hermite_polynomial<T>(1), poly_type{{0, 1}}));
	EXPECT_TRUE(compare(hermite_polynomial<T>(2), poly_type{{-1, 0, 1, 0}}));
}
