#include "apmath/factorial.hh"
#include <gtest/gtest.h>

/// \see http://oeis.org/A000142
TEST(Factorial, Single) {
	using arma::apmath::factorial;
	EXPECT_EQ(factorial(0), 1);
	EXPECT_EQ(factorial(1), 1);
	EXPECT_EQ(factorial(2), 2);
	EXPECT_EQ(factorial(3), 6);
	EXPECT_EQ(factorial(4), 24);
	EXPECT_EQ(factorial(5), 120);
	EXPECT_EQ(factorial(6), 720);
	EXPECT_EQ(factorial(7), 5040);
	EXPECT_EQ(factorial(8), 40320);
	EXPECT_EQ(factorial(9), 362880);
	EXPECT_EQ(factorial(10), 3628800);
}

/// \see http://oeis.org/A006882
TEST(Factorial, Double) {
	using arma::apmath::factorial;
	EXPECT_EQ(factorial(0, 2), 1);
	EXPECT_EQ(factorial(1, 2), 1);
	EXPECT_EQ(factorial(2, 2), 2);
	EXPECT_EQ(factorial(3, 2), 3);
	EXPECT_EQ(factorial(4, 2), 8);
	EXPECT_EQ(factorial(5, 2), 15);
	EXPECT_EQ(factorial(6, 2), 48);
	EXPECT_EQ(factorial(7, 2), 105);
	EXPECT_EQ(factorial(8, 2), 384);
	EXPECT_EQ(factorial(9, 2), 945);
	EXPECT_EQ(factorial(10, 2), 3840);
}

/// \see http://oeis.org/A007661
TEST(Factorial, Triple) {
	using arma::apmath::factorial;
	EXPECT_EQ(factorial(0, 3), 1);
	EXPECT_EQ(factorial(1, 3), 1);
	EXPECT_EQ(factorial(2, 3), 2);
	EXPECT_EQ(factorial(3, 3), 3);
	EXPECT_EQ(factorial(4, 3), 4);
	EXPECT_EQ(factorial(5, 3), 10);
	EXPECT_EQ(factorial(6, 3), 18);
	EXPECT_EQ(factorial(7, 3), 28);
	EXPECT_EQ(factorial(8, 3), 80);
	EXPECT_EQ(factorial(9, 3), 162);
	EXPECT_EQ(factorial(10, 3), 280);
}
