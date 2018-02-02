#include <complex>
#include <fstream>
#include <sstream>

#include <gtest/gtest.h>

#include "apmath/convolution.hh"
#include "domain.hh"
#include "physical_constants.hh"
#include "stats/waves.hh"

typedef ARMA_REAL_TYPE T;

const char* str_slice_x = R"((0,199)
[ 0.81475 0.814963 0.868714 0.694976 0.450768 0.290519 0.00689834 -0.305241 -0.656414 -0.869293 -0.960508 -0.983779 -1.09837 -1.12035 -1.13957 -1.30028 -1.35762 -1.33539 -1.35781 -1.35838 -1.20691 -0.983681 -0.681864 -0.41763 -0.134799 0.250041 0.632535 0.896024 1.0189 1.12609 1.01117 0.913435 0.670272 0.392954 0.207352 0.126609 0.0964246 -0.0342973 -0.0408353 -0.00108368 -0.269243 -0.369182 -0.572463 -0.89234 -1.21686 -1.39032 -1.5386 -1.93338 -2.00957 -1.79692 -1.65749 -1.36972 -1.02718 -0.947096 -0.913806 -0.693509 -0.430349 -0.322982 0.0625449 0.309415 0.472538 0.515416 0.562135 0.719267 0.820082 0.919276 0.954093 1.01716 0.920314 0.853519 0.777454 0.621768 0.515475 0.420502 0.436393 0.404737 0.568875 0.901065 0.928145 1.10412 1.12134 1.04808 0.942881 0.852544 0.845558 1.00012 1.15676 1.21266 1.03833 1.02849 1.06922 1.05315 0.932947 0.838215 0.792264 0.680053 0.572186 0.305733 0.256752 0.159174 0.226257 0.349073 0.200564 0.095174 -0.00975655 -0.0117331 0.140231 -0.0127264 -0.117877 -0.143191 -0.194292 -0.0650367 0.29543 0.722549 1.07049 1.52046 1.9515 2.3189 2.4121 2.53672 2.65692 2.69789 2.76532 2.44049 1.96661 1.68682 1.35341 1.1568 1.0938 1.0265 0.980863 0.832257 0.774616 0.619476 0.363974 0.300962 0.257975 0.198084 0.0486115 -0.187808 -0.379605 -0.519759 -0.567093 -0.603906 -0.58292 -0.670619 -0.725689 -0.610298 -0.583121 -0.766006 -0.924173 -0.901495 -0.944492 -0.944644 -0.946852 -0.777494 -0.624144 -0.640073 -0.618187 -0.725381 -0.408462 -0.14602 -0.0556179 0.224674 0.362937 0.523332 0.598352 0.791822 1.02711 1.04533 0.960286 0.847347 0.639334 0.57129 0.581013 0.592837 0.745391 1.07377 1.15166 1.12387 1.19315 1.13583 1.11671 0.980061 0.719113 0.271758 -0.315417 -0.637489 -0.65251 -0.738762 -1.00197 -0.937109 -0.878852 -0.81577 -0.624746 -0.465241 -0.24782 -0.0446864 -0.0496378 0.128758 ]
)";

TEST(Waves, Features) {
	using arma::constants::_2pi;
	const int n = 101;
	arma::Array1D<T> elevation(n);
	blitz::firstIndex i;
	elevation = blitz::cos(_2pi<T>*2*i/(n-1));
	arma::Grid<T,1> grid({n}, {_2pi<T>*2});
	std::clog << "elevation=" << elevation << std::endl;
	std::clog << "grid=" << grid << std::endl;
	auto features = arma::stats::find_extrema(elevation, grid);
	for (const auto& f : features) {
		std::clog << "f=" << f << std::endl;
	}
	auto waves = arma::stats::find_waves(features);
	for (const auto& w : waves) {
		std::clog << "w=" << w.height() << ',' << w.period() << std::endl;
	}
}

TEST(Waves, FeaturesRealWave) {
	using arma::constants::_2pi;
	arma::Array1D<T> elevation;
	{
		std::stringstream str;
		str << str_slice_x;
		str >> elevation;
	}
	const int n = elevation.numElements();
	arma::Domain<T,1> grid({T(0)}, {T(209.44)}, {n});
	{
		std::ofstream out("elev0");
		for (int i=0; i<n; ++i) {
			out << grid(i,0) << '\t' << elevation(i) << std::endl;
		}
	}
	auto waves0 = arma::stats::find_waves(elevation.copy(), grid, 11);
	arma::stats::smooth_elevation(elevation, grid, 11);
	{
		std::ofstream out("elev");
		for (int i=0; i<n; ++i) {
			out << grid(i,0) << '\t' << elevation(i) << std::endl;
		}
	}
//	std::clog << "elevation=" << elevation << std::endl;
	std::clog << "grid=" << grid << std::endl;
	std::clog << "grid.patch_size(0)=" << grid.patch_size(0) << std::endl;
	auto features = arma::stats::find_extrema(elevation, grid);
	{
		std::ofstream out("extrema");
		for (const auto& f : features) {
			out << f.x << '\t' << f.z << std::endl;
			std::clog << "f=" << f << std::endl;
		}
	}
	auto waves = arma::stats::find_waves(features);
	T avg_height = 0;
	T avg_length = 0;
	for (const auto& w : waves) {
		avg_height += w.height();
		avg_length += w.period();
		std::clog << "w=" << w.height() << ',' << w.period() << std::endl;
	}
	avg_height /= waves.size();
	avg_length /= waves.size();
	std::clog << "avg_height=" << avg_height << std::endl;
	std::clog << "avg_length=" << avg_length << std::endl;
}
