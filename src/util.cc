#include "util.hh"

#include <sstream>
#include <iostream>

void
arma::print_progress(const char* msg, int nfinished, int ntotal) {
	std::stringstream str;
	str << msg << " [" << nfinished << '/' << ntotal << "]\n";
	std::clog << str.rdbuf();
}
