#ifndef DISCRETE_FUNCTION_HH
#define DISCRETE_FUNCTION_HH

#include <blitz/array.h>
#include "grid.hh"

namespace arma {

	/// \brief Multidimensional array with a grid.
	template<class T, int N>
	class Discrete_function: public blitz::Array<T,N> {

		typedef blitz::Array<T,N> base_type;
		typedef Grid<T,N> grid_type;
		grid_type _grid;

	public:
		inline const grid_type&
		grid() const noexcept {
			return this->_grid;
		}

		inline void
		setgrid(const grid_type& rhs) noexcept {
			this->_grid = rhs;
		}

		inline Discrete_function&
		operator=(const Discrete_function& rhs) {
			base_type::operator=(static_cast<const base_type&>(rhs));
			this->_grid = rhs._grid;
			return *this;
		}
	};

}

#endif // DISCRETE_FUNCTION_HH
