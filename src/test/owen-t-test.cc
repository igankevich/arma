#include "apmath/owen_t.hh"
#include <gtest/gtest.h>

typedef ARMA_REAL_TYPE T;
using arma::apmath::owen_t;

TEST(OwenT, Identities) {
	EXPECT_NEAR(owen_t<T>(1, 0), T(0), T(1e-5));
	EXPECT_NEAR(owen_t<T>(1, 1), owen_t<T>(-1, 1), T(1e-5));
	EXPECT_NEAR(owen_t<T>(1, -1), -owen_t<T>(1, 1), T(1e-5));
}

TEST(OwenT, ExactValues) {
	EXPECT_NEAR(owen_t<T>(2, T(0.5)), T(0.00862508), T(1e-5));
	EXPECT_NEAR(owen_t<T>(1, T(1.5)), T(0.07573), T(1e-5));
	EXPECT_NEAR(owen_t<T>(1.5, 2.0), T(0.0333832), T(1e-5));
	EXPECT_NEAR(owen_t<T>(1.0, 0.1), T(0.00960528), T(1e-5));
}
