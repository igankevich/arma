#include <sstream>
#include "output_flags.hh"
#include <gtest/gtest.h>

using arma::Output_flags;

TEST(OutputFlags, SimpleIO) {
	Output_flags f;
	f.setf(Output_flags::ACF);
	f.setf(Output_flags::Surface);
	std::stringstream tmp;
	tmp << f;
	std::string orig = tmp.str();
	Output_flags f2;
	tmp >> f2;
	tmp.clear();
	tmp.str("");
	tmp << f2;
	std::string actual = tmp.str();
	EXPECT_EQ(orig, actual);
}

class OutputFlagsInputTest: public ::testing::TestWithParam<const char*> {};

TEST_P(OutputFlagsInputTest, ReadWhiteSpace) {
	Output_flags orig;
	orig.setf(Output_flags::ACF);
	orig.setf(Output_flags::Surface);
	Output_flags f;
	std::stringstream(GetParam()) >> f;
	EXPECT_EQ(orig, f);
}

INSTANTIATE_TEST_CASE_P(
	ReadWhiteSpace,
	OutputFlagsInputTest,
	::testing::Values(
		"acf,surface",
		"acf , surface",
		" acf , surface ",
		"acf,surface\ntrash"
	)
);
