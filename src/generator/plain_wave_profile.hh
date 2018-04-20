#ifndef GENERATOR_PLAIN_WAVE_PROFILE_HH
#define GENERATOR_PLAIN_WAVE_PROFILE_HH

#include <iosfwd>
#include <ostream>

namespace arma {

	namespace generator {

		enum struct Plain_wave_profile {
			Sine,
			Cosine,
			/// Third-order Stokes wave on deep water
			Stokes,
			Standing_wave,
		};

		std::istream&
		operator>>(std::istream& in, Plain_wave_profile& rhs);

		const char*
		to_string(Plain_wave_profile rhs);

		inline std::ostream&
		operator<<(std::ostream& out, const Plain_wave_profile& rhs) {
			return out << to_string(rhs);
		}

	}

}

#endif // vim:filetype=cpp
