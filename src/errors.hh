#ifndef ERRORS_HH
#define ERRORS_HH

#include <stdexcept>
#include <string>

namespace arma {

	class prng_error: public std::runtime_error {

		int _nprngs, _nparts;

	public:
		prng_error() = default;
		prng_error(const prng_error&) = default;
		prng_error(prng_error&&) = default;

		prng_error(const char* msg, int nprngs, int nparts):
		std::runtime_error(msg),
		_nprngs(nprngs), _nparts(nparts)
		{}

		inline int
		ngenerators() const noexcept {
			return _nprngs;
		}

		inline int
		nparts() const noexcept {
			return _nparts;
		}
	};

}

#endif // ERRORS_HH
