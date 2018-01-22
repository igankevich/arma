#include "util.hh"

#include <sstream>
#include <iostream>

std::mutex arma::__write_mutex;

void
arma::print_progress(const char* msg, int nfinished, int ntotal) {
	std::unique_lock<std::mutex> lock(__write_mutex);
	//std::clog << msg << " [" << nfinished << '/' << ntotal << "]\n";
}
