#include <iostream>
#include <sstream>
#include "domain.hh"

void
test_simple_io() {
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
	if (actual != orig) {
		std::cerr << "Bad Domain i/o: "
			"orig=\"" << orig << "\""
			",actual=\"" << actual << "\""
			<< std::endl;
	}
	assert(actual == orig);
}

void
test_zero_patch() {
	arma::Domain<float, 2> dom{
		{1.f, 2.f},
		{1.f, 10.f},
		{1, 5}
	};
	assert(!(dom.patch_size(0) < 0.f) && !(dom.patch_size(0) < 0.f));
	assert(dom.patch_size(1) > 0.f);
}

int main() {
	test_simple_io();
	test_zero_patch();
	return 0;
}
