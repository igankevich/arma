#ifndef FACTOR_WAVES_HH
#define FACTOR_WAVES_HH

#include <cmath>
#include <vector>

#include "domain.hh"
#include "types.hh"

namespace arma {

	/**
	   \param[in] z  wavy surface
	   \param[in] t  time slice index
	   \param[in] dt \f$\Delta{t}\f$
	   \return wave number range
	 */
	template<class T>
	Domain<T,2>
	factor_waves(Array2D<T> z, int t, T dt);

}

#endif // vim:filetype=cpp
