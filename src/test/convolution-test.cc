#include "apmath/convolution.hh"
#include "types.hh"
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <fstream>
#include <ctime>

using namespace arma;

template <class T>
void
reference_convolve(
	Array3D<T>& zeta,
	Array3D<T>& eps,
	Array3D<T>& kernel
) {
	const Domain3D subdomain = zeta.domain();
	const Shape3D fsize = kernel.shape();
	const Shape3D& lbound = subdomain.lbound();
	const Shape3D& ubound = subdomain.ubound();
	const int t0 = lbound(0);
	const int x0 = lbound(1);
	const int y0 = lbound(2);
	const int t1 = ubound(0);
	const int x1 = ubound(1);
	const int y1 = ubound(2);
	for (int t = t0; t <= t1; t++) {
		for (int x = x0; x <= x1; x++) {
			for (int y = y0; y <= y1; y++) {
				const int m1 = std::min(t + 1, fsize[0]);
				const int m2 = std::min(x + 1, fsize[1]);
				const int m3 = std::min(y + 1, fsize[2]);
				T sum = 0;
				for (int k = 0; k < m1; k++)
					for (int i = 0; i < m2; i++)
						for (int j = 0; j < m3; j++)
							sum += kernel(k, i, j) *
								   eps(t - k, x - i, y - j);
				zeta(t, x, y) = sum;
			}
		}
	}
}

template <class T>
void
reference_convolve(
	Array2D<T>& zeta,
	Array2D<T>& eps,
	Array2D<T>& kernel
) {
	const Domain2D subdomain = zeta.domain();
	const Shape2D fsize = kernel.shape();
	const int nt = eps.extent(0);
	const int nx = eps.extent(1);
	for (int t=0; t<nt; t++) {
		for (int x=0; x<nx; x++) {
			const int m1 = std::min(t + 1, fsize(0));
			const int m2 = std::min(x + 1, fsize(1));
			T sum = 0;
			for (int k = 0; k < m1; k++) {
				for (int i = 0; i < m2; i++) {
					sum += kernel(k, i) * eps(t - k, x - i);
				}
			}
			zeta(t,x) = sum;
		}
	}
}

template <class T>
void
reference_convolve(
	Array1D<T>& zeta,
	Array1D<T>& eps,
	Array1D<T>& kernel
) {
	const Domain1D subdomain = zeta.domain();
	const Shape1D fsize = kernel.shape();
	const int nt = eps.extent(0);
	for (int t=0; t<nt; t++) {
		const int m1 = std::min(t + 1, fsize(0));
		T sum = 0;
		for (int k = 0; k < m1; k++) {
			sum += kernel(k) * eps(t - k);
		}
		zeta(t) = sum;
	}
}

typedef double T;
typedef std::complex<T> C;

template <class X>
void
write_to_files(const X& output, X& actual) {
	using blitz::max;
	using blitz::mean;
	using blitz::abs;
	using blitz::real;
	std::clog << "max error = " << max(abs(real(X(actual - output)))) << std::endl;
	std::clog << "mean error = " << mean(abs(real(X(actual - output)))) << std::endl;
	//std::clog << "diff = " << real(X(actual - output)) << std::endl;
	{
		const T m = max(abs(real(output)));
		std::ofstream out("output");
		std::transform(
			output.begin(),
			output.end(),
			std::ostream_iterator<T>(out, "\n"),
			[m] (const C& rhs) {
				return std::real(rhs) / m;
			}
		);
	}
	{
		const T m = max(abs(real(actual)));
		std::ofstream out("actual");
		std::transform(
			actual.begin(),
			actual.end(),
			std::ostream_iterator<T>(out, "\n"),
			[m] (const C& rhs) {
				return std::real(rhs) / m;
			}
		);
	}
}



TEST(ConvolutionTest, Exceptions) {
	typedef arma::apmath::Convolution<C,1> convolution_type;
	EXPECT_NO_THROW(convolution_type(20, 10));
	EXPECT_THROW(convolution_type(0, 10), std::length_error);
	EXPECT_THROW(convolution_type(-1, 10), std::length_error);
	EXPECT_NO_THROW(convolution_type(20, 0));
	EXPECT_THROW(convolution_type(20, -1), std::length_error);
	EXPECT_THROW(convolution_type(20, 21), std::length_error);
}

/*
TEST(ConvolutionTest, ThreeDim) {
	typedef arma::apmath::Convolution<C,3> convolution_type;
	typedef typename convolution_type::shape_type shape_type;
	typedef typename convolution_type::array_type array_type;
	using blitz::all;
	using blitz::abs;
	shape_type orig(16, 10, 10);
	array_type kernel(orig);
	std::mt19937 prng;
	std::normal_distribution<T> normal(T(0), std::sqrt(T(2)));
	std::generate(kernel.begin(), kernel.end(), std::bind(normal, prng));
	array_type signal(shape_type(200, 200, 200));
	std::generate(signal.begin(), signal.end(), std::bind(normal, prng));
	array_type output(signal.shape());
	reference_convolve(output, signal, kernel);
	convolution_type conv(orig, 0, 100, 15);
	array_type actual(conv.convolve(signal, kernel));
	EXPECT_NEAR(max(abs(actual - output)), 0, 1e-4);
	write_to_files(output, actual);
}
*/

TEST(ConvolutionTest, TwoDim) {
	typedef arma::apmath::Convolution<C,2> convolution_type;
	typedef typename convolution_type::shape_type shape_type;
	typedef typename convolution_type::array_type array_type;
	using blitz::all;
	using blitz::shape;
	using blitz::Range;
	using blitz::abs;
	shape_type orig(16, 16);
	array_type kernel(orig);
	std::mt19937 prng;
	std::normal_distribution<T> normal(T(0), std::sqrt(T(0.1)));
	std::generate(kernel.begin(), kernel.end(), std::bind(normal, prng));
	array_type signal(shape_type(orig(0), orig(1)));
	std::generate(signal.begin(), signal.end(), std::bind(normal, prng));
	array_type output(signal.shape());
	reference_convolve(output, signal, kernel);
	convolution_type conv(shape(orig(0), orig(1)), shape(16,16));
	array_type actual(conv.convolve(signal, kernel));
	write_to_files(output, actual);
	EXPECT_NEAR(max(abs(actual - output)), 0, 1e-4);
}

/*
class Convolution1DTest:
public ::testing::TestWithParam<blitz::TinyVector<int,4>>
{};

TEST_P(Convolution1DTest, OneDim) {
	typedef arma::apmath::Convolution<C,1> convolution_type;
	typedef typename convolution_type::shape_type shape_type;
	typedef typename convolution_type::array_type array_type;
	using blitz::all;
	using blitz::max;
	using blitz::abs;
	auto params = GetParam();
	const int kernel_size = params(0);
	const int signal_size = params(1);
	const int block_size = params(2);
	const int padding_size = params(3);
	shape_type orig(kernel_size);
	array_type kernel(orig);
	std::mt19937 prng;
	std::normal_distribution<T> normal(T(0), std::sqrt(T(2)));
	std::generate(kernel.begin(), kernel.end(), std::bind(normal, prng));
	array_type signal(blitz::shape(signal_size));
	std::generate(signal.begin(), signal.end(), std::bind(normal, prng));
	array_type output(signal.shape());
	reference_convolve(output, signal, kernel);
	convolution_type conv(block_size, padding_size);
	array_type actual(conv.convolve(signal, kernel));
	EXPECT_NEAR(max(abs(actual - output)), 0, 1e-4);
	write_to_files(output, actual);
}

INSTANTIATE_TEST_CASE_P(
	Instance,
	Convolution1DTest,
	::testing::Values(
		blitz::shape(16, 16, 16, 15), // single block
		blitz::shape(16, 1000, 100, 15) // multiple blocks
	)
);
*/
