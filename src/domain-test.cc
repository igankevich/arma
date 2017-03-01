#include <iostream>
#include <sstream>
#include "domain.hh"

int main() {
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
	return 0;
}
