#ifndef VELOCITY_GENERAL_VELOCITY_POTENTIAL_FIELD_HH
#define VELOCITY_GENERAL_VELOCITY_POTENTIAL_FIELD_HH

#include <cmath>
#include <complex>

#include "types.hh"
#include "linear_velocity_potential_field.hh"
#include "fourier.hh"

namespace arma {

	template <class T>
	class High_amplitude_velocity_potential_field:
	public Linear_velocity_potential_field<T> {

	protected:
		/**
		Compute multiplier function as
		\f[
			F(x, y) = -\frac{ \zeta_t }{
				(\zeta_x + \zeta_y) / \sqrt{\smash[b]{1 + \zeta_x^2 + \zeta_y^2}}
				- \zeta_x - \zeta_y - 1
			}
		\f]
		instead of \f$F(x,y)=\zeta_t\f$ as in linear case.
		*/
		void
		precompute(const Array3D<T>& zeta, const int idx_t) override {
			using blitz::Range;
			using blitz::sqrt;
			using blitz::pow;
			Array2D<T> zeta_x(derivative<1,T>(zeta, idx_t));
			Array2D<T> zeta_y(derivative<2,T>(zeta, idx_t));
			Array2D<T> sqrt_zeta(sqrt(T(1) + pow(zeta_x, 2) + pow(zeta_y, 2)));
			this->_zeta_t(idx_t, Range::all(), Range::all()) =
				- derivative<0,T>(zeta, idx_t)
				/ (std::complex<T>(0,1)*((zeta_x + zeta_y)/sqrt_zeta - zeta_x - zeta_y) - T(1));
		}

	};

}

#endif // VELOCITY_GENERAL_VELOCITY_POTENTIAL_FIELD_HH
