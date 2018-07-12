#include <sstream>
#include "fourier.hh"
#include <gtest/gtest.h>

TEST(FourierTest, Shape) {
	typedef std::complex<double> T;
	typedef arma::apmath::Fourier_transform<T,3> fft_type;
	typedef typename fft_type::shape_type shape_type;
	using blitz::all;
	shape_type orig(16, 32, 64);
	fft_type fft(orig);
	EXPECT_TRUE(all(orig == fft.shape()));
}

