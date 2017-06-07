#include "stats/skew_normal.hh"
#include <gtest/gtest.h>

typedef ARMA_REAL_TYPE T;
using arma::stats::Skew_normal;

TEST(SkewNormal, Identities) {
	const T eps = T(1e-5);
	Skew_normal<T> sn(T(0), T(1), T(2));
	EXPECT_NEAR(sn.cdf(-4), T(-1.0842021724855044e-19), eps);
	EXPECT_NEAR(sn.cdf(-3), T(5.596621256709344e-13), eps);
	EXPECT_NEAR(sn.cdf(-2), T(3.143618040497842e-7), eps);
	EXPECT_NEAR(sn.cdf(-1), T(0.0017188799452888537), eps);
	EXPECT_NEAR(sn.cdf(0), T(0.14758361765043326), eps);
	EXPECT_NEAR(sn.cdf(1), T(0.6844083720823747), eps);
	EXPECT_NEAR(sn.cdf(2), T(0.9545000504654456), eps);
	EXPECT_NEAR(sn.cdf(3), T(0.9973002039372995), eps);
	EXPECT_NEAR(sn.cdf(4), T(0.9999366575163338), eps);
}
