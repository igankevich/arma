#ifndef LINEAR_VELOCITY_FIELD_HH
#define LINEAR_VELOCITY_FIELD_HH

#include <cmath>
#include "velocity_potential_field.hh"
#include "fourier.hh"
#include "grid.hh"

namespace arma {

	template<class T>
	class Linear_velocity_potential_field: public Velocity_potential_field<T> {

	public:
		Array2D<T>
		operator()(
			Array3D<T>& zeta,
			const Domain3D& subdomain,
			const T z,
			const int idx_t
		) override {
			Array2D<T> phi(size2(zeta.extent(1), zeta.extent(2)));
			return phi;
		}

	};

}

#endif // LINEAR_VELOCITY_FIELD_HH
