#include <gtest/gtest.h>

#include <chrono>
#include <complex>

#include <blitz/array.h>

#include "arma.hh"
#include "physical_constants.hh"
#include "stats/statistics.hh"


template <class T>
arma::Array3D<T>
auto_covariance_ref(const arma::Array3D<T>& rhs) {
	using namespace arma;
	const Shape3D& shp = rhs.shape();
	const int ni = shp(0);
	const int nj = shp(1);
	const int nk = shp(2);
	const int nall = rhs.numElements();
	Array3D<T> result(rhs.shape());
	#if ARMA_OPENMP
	#pragma omp parallel for collapse(3)
	#endif
	for (int i=0; i<ni; ++i) {
		for (int j=0; j<nj; ++j) {
			for (int k=0; k<nk; ++k) {
				T sum = 0;
				for (int i1=0; i1<ni; ++i1) {
					for (int j1=0; j1<nj; ++j1) {
						for (int k1=0; k1<nk; ++k1) {
							sum += rhs(i1,j1,k1) *
								   rhs((i+i1)%ni,(j+j1)%nj,(k+k1)%nk);
						}
					}
				}
				result(i,j,k) = sum / nall;
			}
		}
	}
	return result;
}

typedef ARMA_REAL_TYPE T;
typedef std::complex<T> C;
typedef std::chrono::high_resolution_clock clock_type;

TEST(AutoCovariance, CosineNoDecay) {
	using arma::constants::_2pi;
	using arma::stats::variance;
	using blitz::abs;
	const T r = T(4.5);
	blitz::TinyVector<int,3> shp(21,21,21);
	if (const char* size = std::getenv("ACF_SIZE")) {
		shp = std::atoi(size);
		std::clog << "shp=" << shp << std::endl;
	}
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
	const auto t0 = clock_type::now();
	blitz::Array<T,3> acf = arma::auto_covariance(wave);
	const auto t1 = clock_type::now();
	blitz::Array<T,3> acf_exp = arma::auto_covariance(wave_exp);
	EXPECT_NEAR(acf(0,0,0), acf_exp(0,0,0), T(1e-3))
		<< "acf(0,0,0)=" << acf(0,0,0) << std::endl
		<< "acf_exp(0,0,0)=" << acf_exp(0,0,0) << std::endl;
	const auto t2 = clock_type::now();
	blitz::Array<T,3> acf_ref = auto_covariance_ref(wave);
	const auto t3 = clock_type::now();
	EXPECT_NEAR(max(abs(acf - acf_ref)), T(0), T(1e-3))
		<< "acf(0,0,0)=" << acf(0,0,0) << std::endl
		<< "acf_ref(0,0,0)=" << acf_ref(0,0,0) << std::endl;
	std::cout
		<< shp(0)
		<< '\t' << (t1-t0).count()
		<< '\t' << (t3-t2).count()
		<< std::endl;
}
