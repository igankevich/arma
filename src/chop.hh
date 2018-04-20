#ifndef CHOP_HH
#define CHOP_HH

#include "types.hh"

namespace arma {

	template <class T>
	Shape3D
	chop_right(const Array3D<T>& rhs, T eps);

}

#endif // vim:filetype=cpp
