#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "generator/ma_coefficient_solver.hh"
#include "blitz.hh"
#include "arma.hh"
#include "domain.hh"

typedef ARMA_REAL_TYPE T;

template <class T>
blitz::Array<T,3>
propagating_wave_ACF(blitz::TinyVector<int,3> shape) {
	using namespace arma;
	using blitz::exp;
	using blitz::cos;
	blitz::firstIndex i;
	blitz::secondIndex j;
	blitz::thirdIndex k;
	T amplitude = 1;
	blitz::TinyVector<T,3> delta(1,1,1);
	blitz::TinyVector<T,3> alpha(1,1,1);
	blitz::TinyVector<T,2> beta(0.4,0.4);
	T velocity = 0.6;
	Array3D<T> acf(shape);
	acf = amplitude *
		exp(-(alpha(0)*i*delta(0) + alpha(1)*j*delta(1) + alpha(2)*k*delta(2))) *
		cos(velocity*i*delta(0) + beta(0)*j*delta(1) + beta(1)*k*delta(2));
	return acf;
}

TEST(MACoefficientSolver, Simple) {
	using namespace arma::generator;
	using namespace arma;
	Shape3D shp(10,10,10);
	Array3D<T> acf = propagating_wave_ACF<T>(shp);
	Shape3D order(5,5,5);
	MA_coefficient_solver<T> solver(acf, order);
	solver();
}
