#ifndef ERRORS_HH
#define ERRORS_HH

#include <stdexcept>
#include <string>

namespace arma {

	/// \brief An error in the number of MT configurations.
	class PRNG_error: public std::runtime_error {

		int _nprngs, _nparts;

	public:
		PRNG_error() = default;
		PRNG_error(const PRNG_error&) = default;
		PRNG_error(PRNG_error&&) = default;

		PRNG_error(const char* msg, int nprngs, int nparts):
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
