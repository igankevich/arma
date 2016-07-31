#include <algorithm>      // for generate_n, copy, for_each
#include <chrono>         // for duration, system_clock, system_clock::time...
#include <functional>     // for reference_wrapper, ref
#include <iostream>       // for cout, operator<<, basi...
#include <iterator>       // for istream_iterator, back_insert_iterator
#include <vector>         // for vector
#include <fstream>        // for ofstream, ifstream
#include <cassert>        // for assert
#include "parallel_mt.hh" // for mt_config, parallel_mt, operator>>, parall...

struct MT_generator {

	MT_generator(const char* filename, size_t nmts)
	    : _filename(filename), _ngenerators(nmts) {}

	void
	generate_and_write_to_file() {
		auto seed = std::chrono::system_clock::now().time_since_epoch().count();
		autoreg::parallel_mt_seq<> seq(seed);
		std::ofstream out(_filename);
		std::ostream_iterator<autoreg::mt_config> out_it(out);
		std::generate_n(out_it, _ngenerators, std::ref(seq));
	}

	void
	read_from_file_and_test() {
		std::ifstream in(_filename);
		std::vector<autoreg::mt_config> params;
		std::copy(std::istream_iterator<autoreg::mt_config>(in),
		          std::istream_iterator<autoreg::mt_config>(),
		          std::back_inserter(params));
		std::cout << "No. of generators = " << params.size() << std::endl;
		assert(params.size() == _ngenerators);
		std::for_each(
		    params.begin(), params.end(), [](autoreg::mt_config& conf) {
			    autoreg::parallel_mt mt(conf);
			    std::generate_n(
			        std::ostream_iterator<autoreg::parallel_mt::result_type>(
			            std::cout, "\n"),
			        3, std::ref(mt));
			});
	}

	const char* _filename;
	size_t _ngenerators;
};

int
main() {
	MT_generator gen("mts.tmp", 7);
	gen.generate_and_write_to_file();
	gen.read_from_file_and_test();
	return 0;
}
