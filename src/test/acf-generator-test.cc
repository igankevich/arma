#include <sstream>

#include <gtest/gtest.h>

#include "generator/acf_generator.hh"

typedef ARMA_REAL_TYPE T;

TEST(ACFGenerator, Simple) {
	std::stringstream input;
	input << R"(
	{
		func = cos
		amplitude = 4
		alpha = (2,1,10)
		velocity = 1
		beta = (0.4,0)
		nwaves = (12.5,12.5,12.5)
		shape = (32,1,1)
	}
)";
	arma::generator::ACF_generator<T> generator;
	input >> generator;
	generator();
}
