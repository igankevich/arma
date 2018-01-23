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
	acf /= acf(0,0,0);
	return acf;
}

template <class T>
blitz::Array<T,3>
propagating_wave_ACF(blitz::TinyVector<int,3> shape) {
	using blitz::exp;
	using blitz::cos;
	blitz::TinyVector<T,3> delta;
	delta = 1;
	blitz::firstIndex i;
	blitz::secondIndex j;
	blitz::thirdIndex k;
	// from mathematica
	T alpha = 0.42, beta = -1.8, gamm = 1.0;
	T velocity = 1.0;
	blitz::Array<T,3> acf(shape);
	acf = gamm*exp(-alpha*(i*delta[0] + j*delta[1] + k*delta[2])) *
		cos(velocity*i*delta[0] + beta*j*delta[1] + 0*beta*k*delta[2]);
	acf /= acf(0,0,0);
	return acf;
}

template <class T, int N>
blitz::Array<T,1>
to_vector(blitz::Array<T,N> rhs) {
	const int n = rhs.numElements();
	blitz::Array<T,1> result(n);
	std::copy_n(rhs.data(), n, result.data());
	return result;
}

template <class T>
blitz::Array<T,3>
solve_yule_walker_gauss_elimination(
	blitz::Array<T,3> acf,
	blitz::TinyVector<int,3> order,
	blitz::Array<T,3> actual
) {
	using blitz::Range;
	using blitz::abs;
	using blitz::max;
	using blitz::shape;
	using blitz::toEnd;
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
	vector_type rhs_copy(rhs.copy());
	linalg::cholesky(lhs, rhs);
	array_type result(order);
	EXPECT_EQ(result.numElements(), rhs.numElements() + 1);
	result(0,0,0) = 0;
	std::copy_n(rhs.data(), rhs.numElements(), result.data() + 1);
	std::clog << "EPS=" << max(abs(linalg::multiply_symmetric_mv(acm, to_vector(result)) - acm(Range::all(),0))) << std::endl;
	std::clog << "EPS_ACTUAL=" << max(abs(linalg::multiply_symmetric_mv(acm, to_vector(actual)) - acm(Range::all(),0))) << std::endl;
	return result;
}

struct YuleWalkerParams {

	typedef ARMA_REAL_TYPE real_type;
	typedef blitz::TinyVector<int,3> shape_type;

	shape_type order;
	real_type variance;
	real_type tolerance;
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
	using blitz::abs;
	using blitz::all;
	using blitz::max;
	using blitz::shape;
	typedef ARMA_REAL_TYPE T;
	const auto& params = GetParam();
	const T variance = params.variance;
	const int order = max(params.order);
	blitz::Array<T,3> acf(params.variance*params.generate_acf(params.order + 1));
	arma::Yule_walker_solver<T> solver(acf);
	solver.max_order(order);
	solver.determine_the_order(false);
	solver.chop(false);
	auto actual = solver.solve();
	auto expected = solve_yule_walker_gauss_elimination(acf, params.order, actual);
	EXPECT_TRUE(all(actual.shape() == expected.shape()))
		<< "order=" << order << std::endl
		<< "actual.shape()=" << actual.shape() << std::endl
		<< "expected.shape()=" << expected.shape() << std::endl;
	EXPECT_NEAR(max(abs(actual - expected)), 0, params.tolerance)
		<< "acf=" << acf << std::endl
		<< "actual=" << actual << std::endl
		<< "expected=" << expected << std::endl;
}

typedef ARMA_REAL_TYPE T;

INSTANTIATE_TEST_CASE_P(
	SquareExponentialACF,
	YuleWalkerTest,
	::testing::Values(
		YuleWalkerParams{{2,2,2}, T(239.2780), T(1e-4), exponential_acf<T>, "exponential_acf"},
		YuleWalkerParams{{3,3,3}, T(239.2780), T(1e-4), exponential_acf<T>, "exponential_acf"},
		YuleWalkerParams{{4,4,4}, T(239.2780), T(1e-4), exponential_acf<T>, "exponential_acf"}
	)
);

INSTANTIATE_TEST_CASE_P(
	SquareStandingWaveACF,
	YuleWalkerTest,
	::testing::Values(
		YuleWalkerParams{{7,7,7}, T(2), T(1e-2), standing_wave_ACF<T>, "standing_wave_ACF"}
	)
);

TEST(YuleWalkerTest, DetermineTheOrder) {
	using blitz::abs;
	using blitz::all;
	using blitz::max;
	using blitz::shape;
	typedef ARMA_REAL_TYPE T;
	YuleWalkerParams params{{10,10,10}, T(239.2780), T(1e-4), exponential_acf<T>, "exponential_acf"};
	const T variance = params.variance;
	blitz::Array<T,3> acf(params.variance*params.generate_acf(params.order + 1));
	arma::Yule_walker_solver<T> solver(acf);
	solver.determine_the_order(true);
	solver.chop(false);
	auto actual = solver.solve();
	EXPECT_TRUE(all(actual.shape() == shape(2,2,2)))
		<< "actual=" << actual << std::endl;
}

TEST(YuleWalkerTest, Chop) {
	using blitz::abs;
	using blitz::all;
	using blitz::max;
	using blitz::shape;
	typedef ARMA_REAL_TYPE T;
	YuleWalkerParams params{{10,10,10}, T(239.2780), T(1e-4), exponential_acf<T>, "exponential_acf"};
	const T variance = params.variance;
	blitz::Array<T,3> acf(params.variance*params.generate_acf(params.order + 1));
	arma::Yule_walker_solver<T> solver(acf);
	solver.determine_the_order(false);
	solver.chop(true);
	auto actual = solver.solve();
	EXPECT_TRUE(all(actual.shape() == shape(2,2,2)))
		<< "actual=" << actual << std::endl;
}

TEST(YuleWalkerTest, RealCase) {
	using blitz::abs;
	using blitz::all;
	using blitz::max;
	using blitz::shape;
	typedef ARMA_REAL_TYPE T;
	YuleWalkerParams params{{10,10,10}, T(2), T(1e-2), standing_wave_ACF<T>, "standing_wave_ACF"};
	const T variance = params.variance;
	blitz::Array<T,3> acf(params.variance*params.generate_acf(params.order + 1));
	arma::Yule_walker_solver<T> solver(acf);
	auto actual = solver.solve();
	EXPECT_EQ(actual.extent(2), 2)
		<< "actual=" << actual << std::endl;
}

