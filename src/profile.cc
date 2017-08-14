#include "profile.hh"
#include <algorithm>
#include <unordered_map>
#include <string>
#include "util.hh"

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

const std::chrono::high_resolution_clock::time_point arma::profile::programme_start =
	std::chrono::high_resolution_clock::now();

void
arma::profile::thread_event(
	const char* name,
	const char* state,
	const char* thread_name,
	const int thread_no
) {
	using namespace std::chrono;
	std::unique_lock<std::mutex> lock(__write_mutex);
	const auto tp = high_resolution_clock::now();
	const auto us = duration_cast<microseconds>(tp - programme_start);
	std::clog << "evnt "
		<< thread_name << ' '
		<< thread_no << ' '
		<< state << ' '
		<< name << ' '
		<< us.count() << "us\n";
}
