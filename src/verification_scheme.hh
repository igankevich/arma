#ifndef VERIFICATION_SCHEME_HH
#define VERIFICATION_SCHEME_HH

#include <istream>
#include <ostream>

namespace arma {

	enum struct Verification_scheme {
		None = 0,
		Summary = 1,
		Quantile = 2,
		Manual = 3,
	};

	std::istream&
	operator>>(std::istream& in, Verification_scheme& rhs);

	const char*
	to_string(Verification_scheme rhs);

	std::ostream&
	operator<<(std::ostream& out, const Verification_scheme& rhs);

}

#endif // VERIFICATION_SCHEME_HH

