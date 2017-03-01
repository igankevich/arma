#ifndef LINEAR_VELOCITY_FIELD_HH
#define LINEAR_VELOCITY_FIELD_HH

#include <cmath>
#include "velocity_potential_field.hh"
#include "fourier.hh"
#include "grid.hh"

namespace arma {

	template<class T>
	class Linear_velocity_potential_field: public Velocity_potential_field<T> {

		Vec2<T> _wnmax;
		T _depth;
		static constexpr const T _2pi = T(2) * M_PI;

	public:

		Array2D<T>
		operator()(
			Array3D<T>& zeta,
			const Domain3D& subdomain,
			const T z,
			const int idx_t
		) override {
			const size3 zeta_size = subdomain.ubound() - subdomain.lbound();
			const size2 arr_size(zeta_size(1), zeta_size(2));
			Array2D<T> phi(arr_size);
			/// 1. Compute wave numbers and multiplier.
			const Grid<T,2> wngrid(arr_size, _wnmax);
			Array2D<T> mult(wngrid.num_points());
			const int nx = wngrid.num_points(0);
			const int ny = wngrid.num_points(1);
			for (int i=0; i<nx; ++i) {
				for (int j=0; j<ny; ++j) {
					const T u = wngrid.delta(i) * i;
					const T v = wngrid.delta(j) * j;
					const T l = _2pi * std::sqrt(u*u + v*v);
					mult(i, j) = std::cosh(l * (z + _depth))
						/ (l * std::cosh(l * _depth));
				}
			}
			/// 2. Compute \f$\zeta_t\f$.
			// TODO Implement it in a separate function with proper handling of borders.
			Array2D<T> zeta_t(arr_size);
			for (int i=1; i<nx-1; ++i) {
				for (int j=1; j<ny-1; ++j) {
					zeta_t(i, j) = zeta(i-1,j) - T(2) * zeta(i,j) + zeta(i+1,j);
				}
			}
			return phi;
		}

		friend std::istream&
		operator>>(std::istream& in, Linear_velocity_potential_field& rhs) {
			sys::parameter_map params({
			    {"wnmax", sys::make_param(rhs._wnmax)},
			    {"depth", sys::make_param(rhs._depth)},
			}, true);
			in >> params;
			return in;
		}

	};

}

#endif // LINEAR_VELOCITY_FIELD_HH
