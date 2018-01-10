#include <gtest/gtest.h>

#include <iostream>

#include "yule_walker.hh"

template <class T>
blitz::Array<T,3>
generate_acf(blitz::TinyVector<int,3> shape) {
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

TEST(YuleWalker, Test1) {
	typedef ARMA_REAL_TYPE T;
	const T variance = 239.2780;
	blitz::Array<T,3> acf = generate_acf<T>(blitz::shape(4,4,4));
	std::clog << "acf=" << acf << std::endl;
	arma::solve_yule_walker(acf, variance, 3);
}
