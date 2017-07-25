#include "apmath/closed_interval.hh"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

typedef ARMA_REAL_TYPE T;
typedef arma::apmath::closed_interval<ARMA_REAL_TYPE> interval_type;
using arma::apmath::closed_interval;

TEST(ClosedInterval, Empty) {
	EXPECT_TRUE(interval_type(1, 0).empty());
	EXPECT_FALSE(interval_type(0, 0).empty());
	EXPECT_FALSE(interval_type(0, 1).empty());
}

TEST(ClosedInterval, IsPoint) {
	const T eps = 1e-3;
	EXPECT_TRUE(interval_type(0, 0).is_point(eps));
	EXPECT_FALSE(interval_type(0, 1).is_point(eps));
	EXPECT_FALSE(interval_type(1, 0).is_point(eps));
}

TEST(ClosedInterval, Valid) {
	EXPECT_TRUE(interval_type(0, 1).valid());
	EXPECT_FALSE(interval_type(0, 0).valid());
	EXPECT_FALSE(interval_type(1, 0).valid());
}
