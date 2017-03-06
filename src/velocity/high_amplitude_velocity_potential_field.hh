#ifndef VELOCITY_GENERAL_VELOCITY_POTENTIAL_FIELD_HH
#define VELOCITY_GENERAL_VELOCITY_POTENTIAL_FIELD_HH

#include <cmath>
#include <complex>

#include "types.hh"
#include "velocity_potential_field.hh"
#include "fourier.hh"

namespace arma {

	template <class T>
	class High_amplitude_velocity_potential_field: public Velocity_potential_field<T> {

		Fourier_transform<std::complex<T>, 2> _fft;
		static constexpr const T _2pi = T(2) * M_PI;

	protected:
		Array2D<T>
		compute_velocity_field_2d(
			Array3D<T>& zeta,
			const size2 arr_size,
			const T z,
			const int idx_t
		) override {
			using blitz::real;
			Array2D<std::complex<T>> phi(arr_size);
			return real(phi);
		}


	};

}

#endif // VELOCITY_GENERAL_VELOCITY_POTENTIAL_FIELD_HH
