#include <sstream>
#include "grid.hh"
#include <gtest/gtest.h>

TEST(GridTest, SimpleIO) {
	typedef arma::Grid<float, 3> grid_type;
	grid_type grid{
		{10, 10, 10},
		{4.0f, 5.0f, 6.0f}
	};
	std::stringstream tmp;
	tmp << grid;
	std::string orig = tmp.str();
	grid_type grid2;
	tmp >> grid2;
	tmp.clear();
	tmp.str("");
	tmp << grid2;
	std::string actual = tmp.str();
	EXPECT_EQ(orig, actual);
}

TEST(GridTest, ZeroPatchSize) {
	arma::Grid<float, 2> grid{
		{1, 5},
		{0.0f, 10.0f}
	};
	EXPECT_FALSE(grid.patch_size(0) < 0.0f);
	EXPECT_FALSE(grid.patch_size(0) > 0.0f);
	EXPECT_GE(grid.patch_size(1), 0.0f);
}


TEST(Array, ViewAssign) {
	using namespace blitz;
	Array<double,2> x(shape(4, 4));
	x = 0;
	Array<double,2> sub(x(Range(2, 3), Range(2, 3)));
	sub = 33;
	std::clog << "x=" << x << std::endl;
	std::clog << "sub=" << sub << std::endl;
}
