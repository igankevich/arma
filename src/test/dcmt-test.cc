#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <vector>
#include <fstream>
#include <gtest/gtest.h>
#include "parallel_mt.hh"

using arma::prng::mt_config;
using arma::prng::parallel_mt;
using arma::prng::parallel_mt_seq;

struct MT_generator {

	MT_generator(const char* filename, size_t nmts):
	_filename(filename), _ngenerators(nmts)
	{}

	void
	generate_and_write_to_file() {
		typedef std::chrono::high_resolution_clock clock_type;
		auto seed = clock_type::now().time_since_epoch().count();
		parallel_mt_seq<> seq(seed);
		std::ofstream out(_filename);
		std::ostream_iterator<mt_config> out_it(out);
		std::generate_n(out_it, _ngenerators, std::ref(seq));
	}

	void
	read_from_file_and_test() {
		std::ifstream in(_filename);
		std::vector<mt_config> params;
		std::copy(std::istream_iterator<mt_config>(in),
		          std::istream_iterator<mt_config>(),
		          std::back_inserter(params));
		std::cout << "No. of generators = " << params.size() << std::endl;
		EXPECT_EQ(_ngenerators, params.size());
		std::for_each(
		    params.begin(), params.end(), [](mt_config& conf) {
			    parallel_mt mt(conf);
			    std::generate_n(
			        std::ostream_iterator<parallel_mt::result_type>(
			            std::cout, "\n"),
			        3, std::ref(mt));
			});
	}

	const char* _filename;
	size_t _ngenerators;
};

TEST(DCMT, IO) {
	MT_generator gen("mts.tmp", 7);
	gen.generate_and_write_to_file();
	gen.read_from_file_and_test();
}
