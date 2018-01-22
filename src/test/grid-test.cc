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
	EXPECT_EQ(1, grid.num_points(0));
	EXPECT_EQ(5, grid.num_points(1));
	EXPECT_EQ(1, grid.num_patches(0));
	EXPECT_EQ(4, grid.num_patches(1));
}

TEST(GridTest, OnePoint) {
	arma::Grid<float, 2> grid{
		{1, 5},
		{2.0f, 10.0f}
	};
	std::clog << "grid.patch_size()=" << grid.patch_size() << std::endl;
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
