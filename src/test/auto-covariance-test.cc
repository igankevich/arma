#include <gtest/gtest.h>

#include <complex>
#include <fstream>

#include <blitz/array.h>

#include "arma.hh"
#include "fourier.hh"
#include "physical_constants.hh"
#include "stats/statistics.hh"

typedef ARMA_REAL_TYPE T;
typedef std::complex<T> C;

TEST(AutoCovariance, CosineNoDecay) {
	using arma::constants::_2pi;
	using arma::stats::variance;
	using blitz::abs;
	const T r = T(4.5);
	blitz::TinyVector<int,3> shp(21,21,21);
	blitz::Array<T,3> wave(shp), wave_exp(shp);
	blitz::firstIndex i;
	blitz::secondIndex j;
	blitz::thirdIndex k;
	blitz::Array<T,1> ts(shp(0)), xs(shp(1)), ys(shp(2));
	ts = -r + T(2)*r*i/(shp(0)-1);
	xs = -r + T(2)*r*i/(shp(1)-1);
	ys = -r + T(2)*r*i/(shp(2)-1);
	wave = T(4)*blitz::cos(_2pi<T>*(xs(i) + 0*ys(j) - ts(k)));
	wave_exp = blitz::exp(-(abs(xs(i)) + abs(ys(j)) + abs(ts(k)))) * wave;
	wave_exp *= std::sqrt(variance(wave) / variance(wave_exp));
	blitz::Array<T,3> acf = arma::auto_covariance(wave);
	blitz::Array<T,3> acf_exp = arma::auto_covariance(wave_exp);
	EXPECT_NEAR(acf(0,0,0), acf_exp(0,0,0), T(1e-3))
		<< "acf(0,0,0)=" << acf(0,0,0) << std::endl
		<< "acf_exp(0,0,0)=" << acf_exp(0,0,0) << std::endl;
	/*
	using arma::apmath::Fourier_transform;
	using arma::apmath::Fourier_workspace;
	Fourier_transform<C,3> fft(wave.shape());
	Fourier_workspace<C,3> workspace = fft.new_workspace();
	blitz::Array<C,3> cwave(wave.shape());
	cwave = wave;
	using blitz::abs;
	using blitz::pow2;
	cwave = pow2(abs(fft.forward(cwave, workspace)));
	cwave = fft.backward(cwave, workspace);
	std::clog << "cwave=" << cwave << std::endl;
	*/
}
