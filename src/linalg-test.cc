#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

#include "linalg.hh"

typedef float T;

TEST(Multiply, IdentityMatrix) {
	using namespace blitz::tensor;
	using blitz::all;
	int n = 10;
	linalg::Matrix<T> lhs(blitz::shape(n, n));
	linalg::Vector<T> rhs(n);
	lhs = (i == j); // identity matrix
	rhs = T(1);
	rhs = linalg::operator*(lhs, rhs);
	EXPECT_TRUE(all(rhs == T(1)));
}

TEST(Multiply, Matrix2) {
	int n = 2;
	linalg::Matrix<T> lhs(blitz::shape(n, n));
	linalg::Vector<T> rhs(n);
	lhs = 1.0, 2.0, 3.0, 4.0;
	rhs = 1.0, 2.0;
	rhs = linalg::operator*(lhs, rhs);
	EXPECT_NEAR(rhs(0), T(5), T(1e-3));
	EXPECT_NEAR(rhs(1), T(11), T(1e-3));
}

TEST(Cholesky, IdentityMatrix) {
	int n = 10;
	linalg::Matrix<T> lhs(blitz::shape(n, n));
	linalg::Vector<T> rhs(n);
	blitz::firstIndex i;
	blitz::secondIndex j;
	lhs = (i == j); // identity matrix
	rhs = T(1);
	linalg::cholesky(lhs, rhs);
	EXPECT_TRUE(std::all_of(rhs.begin(), rhs.end(), [](T x) { return x == T(1); }));
}

TEST(Cholesky, Matrix2) {
	int n = 3;
	linalg::Matrix<T> lhs(blitz::shape(n, n));
	linalg::Vector<T> rhs(n);
	lhs = 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0;
	rhs = T(1);
	linalg::cholesky(lhs, rhs);
	EXPECT_NEAR(rhs(0), T(1), T(1e-3));
	EXPECT_NEAR(rhs(1), T(0), T(1e-3));
	EXPECT_NEAR(rhs(2), T(1), T(1e-3));
}

TEST(Cholesky, Matrix3) {
	int n = 3;
	linalg::Matrix<T> lhs(blitz::shape(n, n));
	linalg::Vector<T> rhs(n);
	lhs = 1.00, 0.50, 0.33, 0.50, 1.00, 0.50, 0.33, 0.50, 1.00;
	rhs = T(1);
	linalg::cholesky(lhs, rhs);
	EXPECT_NEAR(rhs(0), T(0.60241), T(1e-3));
	EXPECT_NEAR(rhs(1), T(0.39759), T(1e-3));
	EXPECT_NEAR(rhs(2), T(0.60241), T(1e-3));
}

TEST(Inverse, IdentityMatrix) {
	using namespace blitz::tensor;
	using blitz::all;
	int n = 10;
	linalg::Matrix<T> lhs(blitz::shape(n, n));
	lhs = (i == j); // identity matrix
	linalg::inverse(lhs);
	EXPECT_TRUE(all(lhs(i,i) == T(1)));
}

TEST(Inverse, Matrix2) {
	int n = 2;
	linalg::Matrix<T> lhs(blitz::shape(n, n));
	lhs = 1.0, 2.0, 3.0, 4.0;
	linalg::inverse(lhs);
	EXPECT_NEAR(lhs(0,0), T(-2), T(1e-3));
	EXPECT_NEAR(lhs(0,1), T(1), T(1e-3));
	EXPECT_NEAR(lhs(1,0), T(1.5), T(1e-3));
	EXPECT_NEAR(lhs(1,1), T(-0.5), T(1e-3));
}
