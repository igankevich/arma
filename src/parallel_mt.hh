#ifndef PARALLEL_MT_HH
#define PARALLEL_MT_HH

#include <cstdint>
#include <limits>
#include <iterator>
#include <algorithm>

#include "dc.h"

namespace autoreg {

	struct mt_config: public ::mt_struct {

		friend std::istream&
		operator>>(std::istream& in, mt_config& rhs) {
		    in.read((char*)&rhs, sizeof(mt_config));
			return in;
		}

	};

	template<int p=521>
	struct parallel_mt_seq {

		typedef mt_config result_type;

		explicit
		parallel_mt_seq(uint32_t seed):
		_seed(seed)
		{}

		result_type
		operator()() {
			this->generate_mt_struct();
			return _result;
		}

		template <class OutputIterator>
		void param(OutputIterator dest) const {
			*dest = _seed; ++dest;
		}

	private:

		void
		generate_mt_struct() {
			mt_config* ptr = ::get_mt_parameter_id_st(nbits, p, _id, _seed);
			if (!ptr) {
				throw std::runtime_error("bad MT");
			}
			_result = *ptr;
			::free_mt_struct(ptr);
			++_id;
		}

		uint32_t _seed = 0;
		uint16_t _id = 0;
		result_type _result = nullptr;

		static const int nbits = 32;

	};

	struct parallel_mt {

		typedef uint32_t result_type;

		explicit
		parallel_mt(mt_config conf):
		_config(conf)
		{ init(0); }

		result_type
		operator()() noexcept {
			return ::genrand_mt(&_config);
		}

		result_type
		min() const noexcept {
			return std::numeric_limits<result_type>::min();
		}

		result_type
		max() const noexcept {
			return std::numeric_limits<result_type>::max();
		}

		void
		seed(result_type rhs) noexcept {
			init(rhs);
		}

	private:

		void
		init(result_type seed) {
			::sgenrand_mt(seed, &_config);
		}

		mt_config _config;

	};

}

#endif // PARALLEL_MT_HH
