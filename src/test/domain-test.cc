#include <sstream>
#include "domain.hh"
#include <gtest/gtest.h>

TEST(DomainTest, SimpleIO) {
	typedef arma::Domain<float, 3> domain_type;
	domain_type dom{
		{1.f, 2.f, 3.f},
		{4.f, 5.f, 6.f},
		{10, 10, 10}
	};
	std::stringstream tmp;
	tmp << dom;
	std::string orig = tmp.str();
	domain_type dom2;
	tmp >> dom2;
	tmp.clear();
	tmp.str("");
	tmp << dom2;
	std::string actual = tmp.str();
	EXPECT_EQ(orig, actual);
}

TEST(DomainTest, ZeroPatchSize) {
	arma::Domain<float, 2> dom{
		{1.f, 2.f},
		{1.f, 10.f},
		{1, 5}
	};
	EXPECT_FALSE(dom.patch_size(0) < 0.f);
	EXPECT_FALSE(dom.patch_size(0) > 0.f);
	EXPECT_GE(dom.patch_size(1), 0.f);
}
