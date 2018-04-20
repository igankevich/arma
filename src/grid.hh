#ifndef GRID_HH
#define GRID_HH

#include <cmath>
#include <istream>

#include "blitz.hh"
#include "validators.hh"

#if ARMA_BSCHEDULER
#include <unistdx/net/pstream>
#endif

/**
\file
\author Ivan Gankevich
\date 2016-07-28
*/

namespace arma {

	/**
	\brief
	A region defined by the number of points and the length along each
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
	template <class T, int N>
	struct Grid {

		typedef blitz::TinyVector<T, N> length_type;
		typedef blitz::TinyVector<int, N> size_type;

		Grid() = default;
		Grid(const Grid&) = default;

		Grid(const size_type& npts, const length_type& len) noexcept:
		_npoints(npts), _length(len)
		{}

		explicit
		Grid(const size_type& npts) noexcept:
		_npoints(npts),
		_length(npts - 1)
		{}

		~Grid() = default;

		inline int
		num_points(int i) const noexcept {
			return this->_npoints(i);
		}

		inline const size_type&
		num_points() const noexcept {
			return this->_npoints;
		}

		inline const size_type&
		size() const noexcept {
			return this->_npoints;
		}

		inline const size_type&
		shape() const noexcept {
			return this->_npoints;
		}

		inline int
		num_patches(int i) const noexcept {
			return std::max(this->_npoints(i) - 1, 1);
		}

		inline size_type
		num_patches() const noexcept {
			return blitz::max(this->_npoints - 1, 1);
		}

		inline T
		patch_size(int i) const noexcept {
			return _length(i) / num_patches(i);
		}

		inline length_type
		patch_size() const noexcept {
			return _length / num_patches();
		}

		inline T
		delta(int i) const noexcept {
			return _length(i) / num_patches(i);
		}

		inline length_type
		delta() const noexcept {
			return _length / num_patches();
		}

		inline T
		length(int i) const noexcept {
			return _length(i);
		}

		inline const length_type&
		length() const noexcept {
			return _length;
		}

		inline T
		ubound(int i) const noexcept {
			return _length(i);
		}

		inline const length_type&
		ubound() const noexcept {
			return _length;
		}

		inline length_type
		operator()(const size_type& i) const noexcept {
			return delta() * i;
		}

		inline T
		operator()(const int idx, const int dim) const noexcept {
			return delta(dim) * idx;
		}

		inline T
		min(const int dim) const noexcept {
			return operator()(0, dim);
		}

		inline length_type
		min() const noexcept {
			return operator()(0);
		}

		inline T
		max(const int dim) const noexcept {
			return operator()(num_patches(), dim);
		}

		inline length_type
		max() const noexcept {
			return operator()(num_patches());
		}

		inline Grid<T,1>
		subgrid(const int dim) const noexcept {
			return Grid<T,1>{{this->_npoints(dim)}, {this->_length(dim)}};
		}

		inline Grid<T,2>
		subgrid(const int dim1, const int dim2) const noexcept {
			return Grid<T,2>{
				{this->_npoints(dim1), this->_npoints(dim2)},
				{this->_length(dim1), this->_length(dim2)}
			};
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

		#if ARMA_BSCHEDULER
		friend sys::pstream&
		operator<<(sys::pstream& out, const Grid& rhs) {
			return out << rhs._npoints << rhs._length;
		}

		friend sys::pstream&
		operator>>(sys::pstream& in, Grid& rhs) {
			return in >> rhs._npoints >> rhs._length;
		}
		#endif

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
