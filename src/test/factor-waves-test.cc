#include <gtest/gtest.h>

#include "stats/waves.hh"
#include "physical_constants.hh"

typedef ARMA_REAL_TYPE T;

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
	auto waves = arma::stats::factor_waves(features);
	for (const auto& w : waves) {
		std::clog << "w=" << w.height() << ',' << w.period() << std::endl;
	}
}
