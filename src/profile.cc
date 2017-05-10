#include "profile.hh"
#include <algorithm>
#include <unordered_map>
#include <string>

arma::counter_type arma::__counters[4096 / sizeof(counter_type)];
std::unordered_map<size_t,std::string> __names;

void
arma::print_counters(std::ostream& out) {
	std::for_each(
		__names.begin(),
		__names.end(),
		[&] (const std::pair<const counter_type,std::string>& rhs) {
			out << "prfl "
				<< rhs.second
				<< " = "
				<< __counters[rhs.first]
				<< "us\n";
		}
	);
}

void
arma::register_counter(size_t idx, std::string name) {
	__names.emplace(idx, name);
}
