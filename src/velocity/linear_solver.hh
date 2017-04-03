#ifndef VELOCITY_LINEAR_SOLVER_HH
#define VELOCITY_LINEAR_SOLVER_HH

#include <complex>
#include "basic_solver.hh"
#include "fourier.hh"

namespace arma {

	namespace velocity {

		/// Linear wave theory formula to compute velocity potential field.
		template<class T>
		class Linear_solver: public Velocity_potential_solver<T> {

		protected:
			typedef std::complex<T> Cmplx;
			Fourier_transform<Cmplx,2> _fft;
			Array3D<Cmplx> _zeta_t;

		protected:
			void
			precompute(const Array3D<T>& zeta) override;

			void
			precompute(const Array3D<T>& zeta, const int idx_t) override;

			Array2D<T>
			compute_velocity_field_2d(
				const Array3D<T>& zeta,
				const Shape2D arr_size,
				const T z,
				const int idx_t
			) override;

		private:
			Array2D<T>
			low_amp_window_function(const Domain<T,2>& wngrid, const T z);

		};

	}
}

#endif // VELOCITY_LINEAR_SOLVER_HH
