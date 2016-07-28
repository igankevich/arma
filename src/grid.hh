#ifndef GRID_HH
#define GRID_HH

#include <iomanip>
#include <istream>
#include <ostream>
#include <blitz/array.h>

namespace autoreg {

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
		num_segments(int i) const {
			return _npoints(i) - 1;
		}

		size_type
		num_segments() const {
			return _npoints - 1;
		}

		T
		patch_size(int i) const {
			return _length(i) / num_segments(i);
		}

		length_type
		patch_size() const {
			return _length / num_segments();
		}

		T
		delta(int i) const {
			return _length(i) / num_segments(i);
		}

		length_type
		delta() const {
			return _length / num_segments();
		}

		T
		length(int i) const {
			return _length(i);
		}

		const length_type&
		length() const {
			return _length;
		}

		friend std::ostream& operator<<(std::ostream& out, const Grid& rhs) {
			return out << rhs._npoints << ':' << rhs._length;
		}

		friend std::istream& operator>>(std::istream& in, Grid& rhs) {
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
}

#endif // GRID_HH
