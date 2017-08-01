#include <iostream>
#include "apmath/convolution.hh"
#include "types.hh"
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <fstream>

using namespace arma;
using blitz::shape;

template <class T>
void
generate_surface(
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
	#if ARMA_OPENMP
	#pragma omp parallel for collapse(3)
	#endif
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
				zeta(t, x, y) = eps(t, x, y) - sum;
			}
		}
	}
}

template <class T>
void
reference_convolve(
	Array3D<T>& zeta,
	Array3D<T>& eps,
	Array3D<T>& kernel
) {
	const Shape3D fsize = kernel.shape();
	const int t1 = zeta.extent(0);
	const int x1 = zeta.extent(1);
	const int y1 = zeta.extent(2);
	#if ARMA_OPENMP
	#pragma omp parallel for collapse(3)
	#endif
	for (int t=0; t<t1; t++) {
		for (int x=0; x<x1; x++) {
			for (int y=0; y<y1; y++) {
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
	#if ARMA_OPENCL
	::arma::opencl::init();
	#endif
	typedef arma::apmath::Convolution<C,1> convolution_type;
	EXPECT_NO_THROW(convolution_type(20, 10));
	EXPECT_THROW(convolution_type(0, 10), std::length_error);
	EXPECT_THROW(convolution_type(-1, 10), std::length_error);
	EXPECT_NO_THROW(convolution_type(20, 0));
	EXPECT_THROW(convolution_type(20, -1), std::length_error);
	EXPECT_THROW(convolution_type(20, 21), std::length_error);
}

template <int N>
struct ConvolutionParams {
	typedef blitz::TinyVector<int,N> shape_type;
	shape_type kernel_size;
	shape_type signal_size;
	shape_type block_size;
	shape_type padding_size;
};

template <int N>
std::ostream&
operator<<(std::ostream& out, const ConvolutionParams<N>& rhs) {
	return out
		<< "kernel_shape=" << rhs.kernel_size
		<< ",signal_shape=" << rhs.signal_size
		<< ",block_shape=" << rhs.block_size
		<< ",padding=" << rhs.padding_size;
}

class GenerateSurfaceTest:
public ::testing::TestWithParam<ConvolutionParams<3>>
{};

#if ARMA_OPENCL
TEST_P(GenerateSurfaceTest, DISABLED_ThreeDim) {
#else
TEST_P(GenerateSurfaceTest, ThreeDim) {
#endif
	#if ARMA_OPENCL
	::arma::opencl::init();
	#endif
	typedef arma::apmath::Convolution<C,3> convolution_type;
	typedef typename convolution_type::array_type array_type;
	using blitz::max;
	using blitz::abs;
	const auto& p = GetParam();
	array_type kernel(p.kernel_size);
	std::mt19937 prng;
	std::normal_distribution<T> normal(T(0), std::sqrt(T(2)));
	std::generate(kernel.begin(), kernel.end(), std::bind(normal, prng));
	array_type signal(p.signal_size);
	std::generate(signal.begin(), signal.end(), std::bind(normal, prng));
	array_type output(signal.shape());
	kernel(0,0,0) = 0;
	generate_surface(output, signal, kernel);
	convolution_type conv(signal, kernel);
	kernel(0,0,0) = -1;
	kernel = -kernel;
	array_type actual(conv.convolve(signal, kernel));
	EXPECT_NEAR(max(abs(actual - output)), 0, 1e-4);
	write_to_files(output, actual);
}

INSTANTIATE_TEST_CASE_P(
	Instance,
	GenerateSurfaceTest,
	::testing::Values(
		ConvolutionParams<3>{
			shape(8,8,8),
			shape(400,8,8),
			shape(100,8,8),
			shape(8,8,8)
		}
	)
);


class Convolution3DTest:
public ::testing::TestWithParam<ConvolutionParams<3>>
{};

#if ARMA_OPENCL
TEST_P(Convolution3DTest, DISABLED_ThreeDim) {
#else
TEST_P(Convolution3DTest, ThreeDim) {
#endif
	#if ARMA_OPENCL
	::arma::opencl::init();
	#endif
	typedef arma::apmath::Convolution<C,3> convolution_type;
	typedef typename convolution_type::array_type array_type;
	using blitz::max;
	using blitz::abs;
	const auto& p = GetParam();
	array_type kernel(p.kernel_size);
	std::mt19937 prng;
	std::normal_distribution<T> normal(T(0), std::sqrt(T(2)));
	std::generate(kernel.begin(), kernel.end(), std::bind(normal, prng));
	array_type signal(p.signal_size);
	std::generate(signal.begin(), signal.end(), std::bind(normal, prng));
	array_type output(signal.shape());
	convolution_type conv(p.block_size, p.padding_size);
	reference_convolve(output, signal, kernel);
	array_type actual(conv.convolve(signal, kernel));
	EXPECT_NEAR(max(abs(actual - output)), 0, 1e-4);
	//write_to_files(output, actual);
}

INSTANTIATE_TEST_CASE_P(
	Instance,
	Convolution3DTest,
	::testing::Values(
		// single block
		ConvolutionParams<3>{
			shape(16,16,16),
			shape(16,16,16),
			shape(16,16,16),
			shape(16,16,16)
		},
		ConvolutionParams<3>{
			shape(8,8,16),
			shape(8,8,16),
			shape(8,8,16),
			shape(8,8,16)
		},
		// multiple blocks
		ConvolutionParams<3>{
			shape(8,8,8),
			shape(400,8,8),
			shape(100,8,8),
			shape(8,8,8)
		}
	)
);

class Convolution2DTest:
public ::testing::TestWithParam<ConvolutionParams<2>>
{};

TEST_P(Convolution2DTest, TwoDim) {
	#if ARMA_OPENCL
	::arma::opencl::init();
	#endif
	typedef arma::apmath::Convolution<C,2> convolution_type;
	typedef typename convolution_type::array_type array_type;
	using blitz::abs;
	using blitz::max;
	const auto& p = GetParam();
	array_type kernel(p.kernel_size);
	std::mt19937 prng;
	std::normal_distribution<T> normal(T(0), std::sqrt(T(0.1)));
	std::generate(kernel.begin(), kernel.end(), std::bind(normal, prng));
	array_type signal(p.signal_size);
	std::generate(signal.begin(), signal.end(), std::bind(normal, prng));
	array_type output(signal.shape());
	reference_convolve(output, signal, kernel);
	convolution_type conv(p.block_size, p.padding_size);
	array_type actual(conv.convolve(signal, kernel));
	//write_to_files(output, actual);
	EXPECT_NEAR(max(abs(actual - output)), 0, 1e-4);
}

INSTANTIATE_TEST_CASE_P(
	Instance,
	Convolution2DTest,
	::testing::Values(
		// single block
		ConvolutionParams<2>{shape(16,16), shape(16,16), shape(16,16), shape(16,16)},
		ConvolutionParams<2>{shape(8,16), shape(8,16), shape(8,16), shape(8,16)},
		ConvolutionParams<2>{shape(16,8), shape(16,8), shape(16,8), shape(16,8)},
		// multiple blocks
		ConvolutionParams<2>{shape(16,16), shape(1000,16), shape(100,16), shape(16,16)}
	)
);

class Convolution1DTest:
public ::testing::TestWithParam<blitz::TinyVector<int,4>>
{};

TEST_P(Convolution1DTest, OneDim) {
	#if ARMA_OPENCL
	::arma::opencl::init();
	#endif
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
	array_type signal(shape(signal_size));
	std::generate(signal.begin(), signal.end(), std::bind(normal, prng));
	array_type output(signal.shape());
	reference_convolve(output, signal, kernel);
	convolution_type conv(block_size, padding_size);
	array_type actual(conv.convolve(signal, kernel));
	EXPECT_NEAR(max(abs(actual - output)), 0, 1e-4);
	//write_to_files(output, actual);
}

INSTANTIATE_TEST_CASE_P(
	Instance,
	Convolution1DTest,
	::testing::Values(
		shape(16, 16, 16, 15), // single block
		shape(16, 1000, 100, 15) // multiple blocks
	)
);
