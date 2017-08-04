#ifndef PARALLEL_MT_HH
#define PARALLEL_MT_HH

#include <cstdlib>   // for free, malloc
#include <algorithm> // for copy_n
#include <cstdint>   // for uint32_t, uint16_t
#include <cstring>   // for memset, memcpy
#include <istream>   // for istream, ostream, basic_istream::read, basic_os...
#include <limits>    // for numeric_limits
#include <stdexcept> // for runtime_error
#include <vector>
#include <fstream>
#include <iterator>
#include <chrono>

extern "C" {
#include <dc.h> // for free_mt_struct, genrand_mt, get_mt_parameter_id_st
}

namespace arma {

	/// \brief Pseudo-random number generators.
	namespace prng {

		/// \brief A set of parameters for parallel_mt generator.
		struct mt_config : public ::mt_struct {

			mt_config() { std::memset(this, 0, sizeof(mt_config)); }
			~mt_config() { std::free(this->state); }
			mt_config(const mt_config& rhs) {
				std::memset(this, 0, sizeof(mt_config));
				this->operator=(rhs);
			}

			mt_config&
			operator=(const mt_config& rhs) {
				free_state();
				std::memcpy(this, &rhs, sizeof(mt_config));
				init_state();
				std::copy_n(rhs.state, rhs.nn, this->state);
				return *this;
			}

		private:
			void
			init_state() {
				this->state = (uint32_t*)std::malloc(this->nn * sizeof(uint32_t));
			}

			void
			free_state() {
				if (this->state) { std::free(this->state); }
			}

			friend std::istream&
			operator>>(std::istream& in, mt_config& rhs) {
				rhs.free_state();
				in.read((char*)&rhs, sizeof(mt_config));
				rhs.init_state();
				in.read((char*)rhs.state, rhs.nn * sizeof(uint32_t));
				return in;
			}

			friend std::ostream&
			operator<<(std::ostream& out, const mt_config& rhs) {
				out.write((char*)&rhs, sizeof(mt_config));
				out.write((char*)rhs.state, rhs.nn * sizeof(uint32_t));
				return out;
			}
		};

		/**
		\brief Generates a sequence of mt_config objects.

		mt_config objects are used to initialise parallel_mt generators to make
		them produce uncorrelated sequences of pseudo-random numbers in
		parallel.
		*/
		template <int p = 521, int w = 32>
		struct parallel_mt_seq {

			typedef mt_config result_type;

			explicit parallel_mt_seq(uint32_t seed) : _seed(seed) {}

			result_type operator()() {
				this->generate_mt_struct();
				return _result;
			}

			template <class OutputIterator>
			void
			param(OutputIterator dest) const {
				*dest = _seed;
				++dest;
			}

		private:
			void
			generate_mt_struct() {
				mt_config* ptr = static_cast<mt_config*>(
				    ::get_mt_parameter_id_st(w, p, _id, _seed));
				if (!ptr) { throw std::runtime_error("bad MT"); }
				_result = *ptr;
				::free_mt_struct(ptr);
				++_id;
			}

			uint32_t _seed = 0;
			uint16_t _id = 0;
			result_type _result;
		};

		/**
		\brief A version of Mersenne Twister which is able to produce
		uncorrelated pseudo-random number sequences using pre-generated
		mt_config objects.
		*/
		struct parallel_mt {

			typedef uint32_t result_type;

			parallel_mt() = default;

			explicit
			parallel_mt(mt_config conf):
			parallel_mt(conf, 0)
			{}

			parallel_mt(mt_config conf, result_type seed):
			_config(conf)
			{ init(seed); }

			parallel_mt&
			operator=(const parallel_mt&) = default;

			result_type
			operator()() noexcept { return ::genrand_mt(&_config); }

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

		template<class Result>
		void
		read_parallel_mt_config(const char* filename, Result result) {
			std::ifstream in(filename);
			if (!in.is_open()) {
				throw std::runtime_error("bad file");
			}
			std::copy(
				std::istream_iterator<mt_config>(in),
				std::istream_iterator<mt_config>(),
				result
			);
		}

		typedef std::chrono::high_resolution_clock clock_type;

		inline clock_type::rep
		clock_seed() noexcept {
			return clock_type::now().time_since_epoch().count();
		}

		std::vector<parallel_mt>
		read_parallel_mts(const char* filename, size_t n, bool noseed);

	}

}

#endif // PARALLEL_MT_HH
