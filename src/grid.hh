#ifndef GRID_HH
#define GRID_HH

#include <stddef.h>      // for size_t
#include <istream>       // for istream, ostream, basic_istream::putback
#include <blitz/array.h> // for TinyVector
#include "validators.hh"

/**
\file
\author Ivan Gankevich
\date 2016-07-28
*/

namespace arma {

	/**
	\brief
	Grid is defined by the number of points and the length along each
	dimension.

	\details
	The number of points equals the number of patches plus 1. If the length is
	omitted then it is automatically set to make patch length equal to 1.
	For example, the following grid has 4 points and 3 patches.
	\dot
	    graph {
	        graph [rankdir="LR"];
	        node [label="",shape="circle",width=0.23];
	        a--b--c--d;
	    }
	\enddot

	\tparam T length type.
	\tparam N no. of dimensions.
	*/
	template <class T, size_t N>
	struct Grid {

		typedef blitz::TinyVector<T, N> length_type;
		typedef blitz::TinyVector<int, N> size_type;

		Grid() = default;
		Grid(const Grid&) = default;
		Grid(const size_type& npts, const length_type& len)
		    : _npoints(npts), _length(len) {}
		explicit Grid(const size_type& npts)
		    : _npoints(npts), _length(npts - 1) {}
		~Grid() = default;

		int
		num_points(int i) const {
			return _npoints(i);
		}

		const size_type&
		num_points() const {
			return _npoints;
		}

		const size_type&
		size() const {
			return _npoints;
		}

		int
		num_patches(int i) const {
			return _npoints(i) - 1;
		}

		size_type
		num_patches() const {
			return _npoints - 1;
		}

		T
		patch_size(int i) const {
			return _length(i) / num_patches(i);
		}

		length_type
		patch_size() const {
			return _length / num_patches();
		}

		T
		delta(int i) const {
			return _length(i) / num_patches(i);
		}

		length_type
		delta() const {
			return _length / num_patches();
		}

		T
		length(int i) const {
			return _length(i);
		}

		const length_type&
		length() const {
			return _length;
		}

		friend std::ostream&
		operator<<(std::ostream& out, const Grid& rhs) {
			return out << rhs._npoints << ':' << rhs._length;
		}

		friend std::istream&
		operator>>(std::istream& in, Grid& rhs) {
			char delim;
			if (in >> rhs._npoints >> std::ws >> delim) {
				if (delim == ':') {
					in >> rhs._length;
				} else {
					in.putback(delim);
					rhs._length = rhs._npoints - 1;
				}
			}
			return in;
		}

	private:
		size_type _npoints;
		length_type _length;
	};

	template<class T, int n>
	void
	validate_grid(const Grid<T,n>& rhs, const char* name) {
		validate_shape(rhs.num_points(), name);
		validate_shape(rhs.length(), name);
		validate_finite(rhs.length(), name);
	}

}

#endif // GRID_HH
