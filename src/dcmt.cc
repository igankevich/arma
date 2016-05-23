#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <functional>
#include <vector>

#include "parallel_mt.hh"

void
generate_mersenne_twisters() {
	auto seed = std::chrono::system_clock::now().time_since_epoch().count();
	autoreg::parallel_mt_seq<> seq(seed);
	std::ofstream out("MersenneTwister.dat");
	std::generate_n(
		std::ostream_iterator<autoreg::mt_config>(out),
		10, std::ref(seq));

}

int main() {
	generate_mersenne_twisters();
	std::ifstream in("MersenneTwister.dat");
	std::vector<autoreg::mt_config> params;
	std::copy(
		std::istream_iterator<autoreg::mt_config>(in),
		std::istream_iterator<autoreg::mt_config>(),
		std::back_inserter(params)
	);
	std::cout << "No. of generators = " << params.size() << std::endl;
	std::for_each(params.begin(), params.end(),
		[] (autoreg::mt_config& conf) {
			autoreg::parallel_mt mt(conf);
			std::generate_n(
				std::ostream_iterator<autoreg::parallel_mt::result_type>(std::cout, "\n"),
				3, std::ref(mt));
		}
	);
	return 0;
}
