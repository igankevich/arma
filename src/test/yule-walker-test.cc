#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <string>

#include "generator/voodoo.hh"
#include "linalg.hh"
#include "yule_walker.hh"

template <class T>
blitz::Array<T,3>
exponential_acf(blitz::TinyVector<int,3> shape) {
	blitz::firstIndex a;
	blitz::secondIndex b;
	blitz::thirdIndex c;
	blitz::Array<T,3> result(shape);
	result =
		blitz::pow(T(0.9), a) *
		blitz::pow(T(0.88), b) *
		blitz::pow(T(0.95), c);
	return result;
}

template <class T>
blitz::Array<T,3>
standing_wave_ACF(blitz::TinyVector<int,3> shape) {
	using blitz::exp;
	using blitz::cos;
	blitz::TinyVector<T,3> delta;
	delta = 1;
	blitz::firstIndex t;
	blitz::secondIndex x;
	blitz::thirdIndex y;
	T alpha = 0.06;
	T beta = 0.8;
	T gamm = 1.0;
	T velocity = 2*beta;
	blitz::Array<T,3> acf(shape);
	acf = gamm *
		exp(-alpha*(2*t*delta[0] + x*delta[1] + y*delta[2])) *
		cos(velocity*t*delta[0]) *
		cos(beta*x*delta[1]) *
		cos(0*beta*y*delta[2]);
	return acf;
}

template <class T>
blitz::Array<T,3>
solve_yule_walker_gauss_elimination(blitz::Array<T,3> acf, int order) {
	using blitz::Range;
	using blitz::toEnd;
	using blitz::shape;
	typedef blitz::Array<T,3> array_type;
	typedef blitz::Array<T,2> matrix_type;
	typedef blitz::Array<T,1> vector_type;
	arma::generator::AC_matrix_generator<T> gen(acf, order);
	matrix_type acm(gen());
	const int m = acm.rows() - 1;
	// eliminate the first equation and move the first column of the
	// remaining matrix to the right-hand side of the system
	vector_type rhs(m);
	rhs = acm(Range(1, toEnd), 0);
	// lhs is the autocovariance matrix without first
	// column and row
	matrix_type lhs(shape(m, m));
	lhs = acm(Range(1, toEnd), Range(1, toEnd));
	EXPECT_EQ(lhs.extent(0), m);
	EXPECT_EQ(lhs.extent(1), m);
	EXPECT_EQ(rhs.extent(0), m);
	EXPECT_TRUE(linalg::is_symmetric(lhs));
	EXPECT_TRUE(linalg::is_positive_definite(lhs));
	linalg::cholesky(lhs, rhs);
	array_type result(shape(order, order, order));
	EXPECT_EQ(result.numElements(), rhs.numElements() + 1);
	result(0,0,0) = 0;
	std::copy_n(rhs.data(), rhs.numElements(), result.data() + 1);
	return result;
}

struct YuleWalkerParams {

	typedef ARMA_REAL_TYPE real_type;
	typedef blitz::TinyVector<int,3> shape_type;

	shape_type order;
	std::function<decltype(exponential_acf<real_type>)> generate_acf;
	std::string name;
};

std::ostream&
operator<<(std::ostream& out, const YuleWalkerParams& rhs) {
	return out << "order=" << rhs.order << ",acf=" << rhs.name;
}

class YuleWalkerTest:
public ::testing::TestWithParam<YuleWalkerParams>
{};

TEST_P(YuleWalkerTest, CompareToGaussElimination) {
	using blitz::max;
	using blitz::abs;
	using blitz::shape;
	typedef ARMA_REAL_TYPE T;
	const T variance = 239.2780;
	const auto& params = GetParam();
	const int order = max(params.order);
	auto acf = exponential_acf<T>(params.order + 1);
	auto actual = arma::solve_yule_walker(acf, variance, order);
	auto expected = solve_yule_walker_gauss_elimination(acf, order);
	EXPECT_NEAR(max(abs(actual - expected)), 0, 1e-4)
		<< "acf=" << acf << std::endl
		<< "actual=" << actual << std::endl
		<< "expected=" << expected << std::endl;
}

INSTANTIATE_TEST_CASE_P(
	SquareExponentialACF,
	YuleWalkerTest,
	::testing::Values(
		YuleWalkerParams{{2,2,2}, exponential_acf<ARMA_REAL_TYPE>, "exponential_acf"},
		YuleWalkerParams{{3,3,3}, exponential_acf<ARMA_REAL_TYPE>, "exponential_acf"},
		YuleWalkerParams{{4,4,4}, exponential_acf<ARMA_REAL_TYPE>, "exponential_acf"}
	)
);

INSTANTIATE_TEST_CASE_P(
	SquareStandingWaveACF,
	YuleWalkerTest,
	::testing::Values(
		YuleWalkerParams{{2,2,2}, standing_wave_ACF<ARMA_REAL_TYPE>, "standing_wave_ACF"},
		YuleWalkerParams{{3,3,3}, standing_wave_ACF<ARMA_REAL_TYPE>, "standing_wave_ACF"},
		YuleWalkerParams{{4,4,4}, standing_wave_ACF<ARMA_REAL_TYPE>, "standing_wave_ACF"},
		YuleWalkerParams{{7,7,7}, standing_wave_ACF<ARMA_REAL_TYPE>, "standing_wave_ACF"}
	)
);


