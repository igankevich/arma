#ifndef MA_ALGORITHM_HH
#define MA_ALGORITHM_HH

#include <istream>
#include <ostream>

namespace arma {

	enum struct MA_algorithm {
		Fixed_point_iteration = 0,
		Newton_Raphson = 1,
	};

	std::istream&
	operator>>(std::istream& in, MA_algorithm& rhs);

	std::ostream&
	operator<<(std::ostream& out, const MA_algorithm& rhs);

	const char*
	to_string(MA_algorithm rhs);

}

#endif // MA_ALGORITHM_HH
