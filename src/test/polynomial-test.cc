#include "apmath/polynomial.hh"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include "polynomial.hh"

typedef ARMA_REAL_TYPE T;
typedef arma::apmath::Polynomial<ARMA_REAL_TYPE> poly_type;

TEST(Polynomial, Constructor) {
	EXPECT_EQ(poly_type().order(), 0);
	EXPECT_EQ(poly_type(1).order(), 1);
	EXPECT_EQ(poly_type(2).order(), 2);
}

TEST(Polynomial, Assign) {
	poly_type p1{{1, 2, 3, 4}};
	poly_type p2;
	p2 = p1;
	EXPECT_TRUE(std::equal(&p1[0], &p1[p1.order()], &p2[0]));
}

TEST(Polynomial, Multiply) {
	poly_type p1{{1, 2}};
	poly_type p2{{3, 4}};
	poly_type p3{{3, 10, 8}};
	p1 = p1*p2;
	EXPECT_TRUE(std::equal(
		&p1[0],
		&p1[p1.order()],
		&p3[0],
		[] (T a, T b) { return std::abs(a-b) < T(1e-3); }
	))
		<< "p1=" << p1 << std::endl
		<< "p2=" << p2 << std::endl
		<< "p1*p2=" << p1 << std::endl;
}

TEST(Polynomial, Normalise) {
	const T eps(1e-3);
	EXPECT_TRUE(compare(poly_type({0}).normalise(eps), poly_type()));
	EXPECT_TRUE(compare(poly_type({0, 0}).normalise(eps), poly_type()));
	EXPECT_TRUE(compare(poly_type({0, 0, 0}).normalise(eps), poly_type()));
}
