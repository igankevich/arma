#ifndef YULE_WALKER_HH
#define YULE_WALKER_HH

#include "types.hh"

namespace arma {

	/**
	\brief Compute AR model coefficients using an order-recursvive method
	from \cite ByoungSeon1999.

	The AR model order is determined automatically.
	*/
	template <class T>
	void
	solve_yule_walker(Array3D<T> acf, const T variance, const int max_order);

}

#endif // vim:filetype=cpp
