#ifndef GENERATOR_AR_ALGORITHM_HH
#define GENERATOR_AR_ALGORITHM_HH

#include <istream>
#include <ostream>

namespace arma {

	/**
	\brief The algorithm for computing autoregressive model coefficients.
	\date 2018-01-12
	\author Ivan Gankevich
	*/
	enum struct AR_algorithm {
		Gauss_elimination = 0,
		Choi = 1,
	};

	std::istream&
	operator>>(std::istream& in, AR_algorithm& rhs);

	std::ostream&
	operator<<(std::ostream& out, const AR_algorithm& rhs);

	const char*
	to_string(AR_algorithm rhs);

}

#endif // vim:filetype=cpp

