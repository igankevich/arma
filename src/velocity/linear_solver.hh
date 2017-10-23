#ifndef VELOCITY_LINEAR_SOLVER_HH
#define VELOCITY_LINEAR_SOLVER_HH

#include <complex>

#include "basic_solver.hh"
#include "fourier.hh"

namespace arma {

	namespace velocity {

		/**
		\brief Uses linear wave theory formula, small-amplitude waves.
		\ingroup solvers
		*/
		template<class T>
		class Linear_solver: public Velocity_potential_solver<T> {

		protected:
			using typename Velocity_potential_solver<T>::domain2_type;
			typedef std::complex<T> Cmplx;
			typedef apmath::Fourier_transform<Cmplx,2> transform_type;
			typedef apmath::Fourier_workspace<Cmplx,2> workspace_type;

		protected:
			transform_type _fft;
			Array3D<Cmplx> _zeta_t;
			#if ARMA_DEBUG_FFT
			Array3D<T> _wnfunc;
			Array3D<Cmplx> _fft_1;
			int _idxz = 0;
			#endif

		public:
			inline
			Linear_solver():
			Velocity_potential_solver<T>(),
			_fft()
			{}

		protected:
			void
			precompute(const Discrete_function<T,3>& zeta) override;

			void
			precompute(const Discrete_function<T,3>& zeta, const int idx_t) override;

			/**
			\param[in] zeta      ocean wavy surface
			\param[in] arr_size  zeta extent in \f$Y\f$ and \f$Y\f$ dimensions
			\param[in] z         a coordinate \f$z\f$ in which to compute velocity
								 potential
			\param[in] idx_t     a time point in which to compute velocity potential,
								 specified as index of zeta
			*/
			Array2D<T>
			compute_velocity_field_2d(
				const Discrete_function<T,3>& zeta,
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
