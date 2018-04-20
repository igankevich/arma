#ifndef DISCRETE_FUNCTION_HH
#define DISCRETE_FUNCTION_HH

#if ARMA_BSCHEDULER
#include <unistdx/net/pstream>
#endif

#include "grid.hh"
#include "types.hh"

namespace arma {

	/// \brief Multidimensional array with a grid.
	template<class T, int N>
	class Discrete_function: public ::arma::Array<T,N> {

		typedef ::arma::Array<T,N> base_type;
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

		inline void
		reference(const base_type& rhs) {
			base_type::reference(static_cast<const base_type&>(rhs));
		}

		inline void
		reference(const Discrete_function& rhs) {
			base_type::reference(static_cast<const base_type&>(rhs));
			this->_grid = rhs._grid;
		}

		#if ARMA_BSCHEDULER
		inline friend sys::pstream&
		operator<<(sys::pstream& out, const Discrete_function& rhs) {
			out << static_cast<const base_type&>(rhs);
			out << rhs._grid;
			return out;
		}

		inline friend sys::pstream&
		operator>>(sys::pstream& in, Discrete_function& rhs) {
			in >> static_cast<base_type&>(rhs);
			in >> rhs._grid;
			return in;
		}

		#endif

	};

}

#endif // DISCRETE_FUNCTION_HH
