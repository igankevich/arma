#ifndef DOMAIN_HH
#define DOMAIN_HH

#include <cmath>
#include <iomanip>
#include <istream>
#include <ostream>

#include <blitz/array.h>

#if ARMA_BSCHEDULER
#include <unistdx/net/pstream>
#endif

#include "grid.hh"
#include "validators.hh"

namespace arma {

	/**
	\brief
	A region defined by the number of points and the lower and upper bound
	for each dimension.

	\tparam T length type.
	\tparam N no. of dimensions.
	*/
	template <class T, int N=1>
	struct Domain {

		typedef blitz::TinyVector<T, N> length_type;
		typedef blitz::TinyVector<int, N> size_type;

		Domain() = default;
		Domain(const Domain&) = default;
		Domain(Domain&&) = default;

		inline
		Domain(
			const length_type& lbound,
			const length_type& ubound,
			const size_type& npts
		): _lbound(lbound), _ubound(ubound), _npoints(npts)
		{}

		inline
		Domain(const length_type& ubound, const size_type& npts):
		_lbound(), _ubound(ubound), _npoints(npts)
		{}

		inline
		Domain(const Domain& d, const size_type& npts):
		_lbound(d._lbound), _ubound(d._ubound), _npoints(npts)
		{}

		explicit
		Domain(const size_type& npts):
		_npoints(npts),
		_lbound(),
		_ubound(npts - 1)
		{}

		explicit
		Domain(const Grid<T,N>& rhs):
		_lbound(),
		_ubound(rhs.length()),
		_npoints(rhs.num_points())
		{}

		~Domain() = default;

		Domain&
		operator=(const Domain&) = default;

		Domain&
		operator=(const Grid<T,N>& rhs) {
			this->_lbound = T(0);
			this->_ubound = rhs.ubound();
			this->_npoints = rhs.num_points();
			return *this;
		}

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

		const size_type&
		shape() const {
			return _npoints;
		}

		const length_type&
		lbound() const {
			return _lbound;
		}

		T
		lbound(int i) const noexcept {
			return _lbound(i);
		}

		const length_type&
		ubound() const {
			return _ubound;
		}

		T
		ubound(int i) const noexcept {
			return _ubound(i);
		}

		int
		num_patches(int i) const {
			return std::max(this->_npoints(i) - 1, 1);
		}

		size_type
		num_patches() const {
			return blitz::max(this->_npoints - 1, 1);
		}

		length_type
		length() const noexcept {
			return _ubound - _lbound;
		}

		T
		length(int i) const noexcept {
			return _ubound(i) - _lbound(i);
		}

		T
		patch_size(int i) const {
			const int npatches = num_patches(i);
			return npatches == 0 ? T(0) : (length(i) / npatches);
		}

		length_type
		patch_size() const {
			const size_type npatches = num_patches();
			return blitz::where(npatches == 0, T(0), length() / npatches);
		}

		T
		delta(int i) const {
			return patch_size(i);
		}

		length_type
		delta() const {
			return patch_size();
		}

		length_type
		operator()(const size_type& i) const noexcept {
			return _lbound + delta() * i;
		}

		T
		operator()(const int idx, const int dim) const noexcept {
			return _lbound(dim) + delta(dim) * idx;
		}

		/**
		\brief
		Translate lower and upper domain bounds by specified number of points.
		\date 2018-02-02
		\author Ivan Gankevich
		*/
		inline void
		translate(const size_type& shift) {
			const length_type d(shift*delta());
			this->_lbound += d;
			this->_ubound += d;
		}

		friend std::ostream&
		operator<<(std::ostream& out, const Domain& rhs) {
			return out << "from" << ' ' << rhs._lbound << ' '
				<< "to" << ' ' << rhs._ubound << ' '
				<< "npoints" << ' ' << rhs._npoints;
		}

		friend std::istream&
		operator>>(std::istream& in, Domain& rhs) {
			rhs._lbound = T(0);
			rhs._ubound = T(0);
			rhs._npoints = 1;
			int ntokens = 0;
			std::string prefix;
			while (ntokens < 3 && in >> std::ws >> prefix) {
				++ntokens;
				if (prefix == "from") {
					in >> rhs._lbound;
				} else if (prefix == "to") {
					in >> rhs._ubound;
				} else if (prefix == "npoints") {
					in >> rhs._npoints;
				} else {
					std::cerr << "bad prefix: " << prefix << std::endl;
					--ntokens;
					in.setstate(std::ios_base::failbit);
				}
			}
			return in;
		}

		#if ARMA_BSCHEDULER
		friend sys::pstream&
		operator<<(sys::pstream& out, const Domain& rhs) {
			return out << rhs._lbound << rhs._ubound << rhs._npoints;
		}

		friend sys::pstream&
		operator>>(sys::pstream& in, Domain& rhs) {
			return in >> rhs._lbound >> rhs._ubound >> rhs._npoints;
		}
		#endif

	private:
		length_type _lbound;
		length_type _ubound;
		size_type _npoints;
	};

	template<class T, int n>
	void
	validate_domain(const Domain<T,n>& rhs, const char* name) {
		validate_shape(rhs.num_points(), name);
		validate_finite(rhs.lbound(), name);
		validate_finite(rhs.ubound(), name);
	}

	template<class T> using Domain2 = Domain<T,2>;
	template<class T> using Domain3 = Domain<T,3>;

}

#endif // DOMAIN_HH
