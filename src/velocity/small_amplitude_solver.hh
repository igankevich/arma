#ifndef VELOCITY_SMALL_AMPLITUDE_SOLVER_HH
#define VELOCITY_SMALL_AMPLITUDE_SOLVER_HH

#include "basic_solver.hh"

namespace arma {

	namespace velocity {

		/// \note WIP
		template <class T>
		class Small_amplitude_solver: public Velocity_potential_solver<T> {

		protected:
			Array2D<T>
			compute_velocity_field_2d(
				const Discrete_function<T,3>& zeta,
				const Shape2D arr_size,
				const T z,
				const int idx_t
			) override;

			void
			precompute(const Discrete_function<T,3>& zeta) override {}

			void
			precompute(const Discrete_function<T,3>& zeta, const int idx_t) override {}
		};

	}

}

#endif // VELOCITY_SMALL_AMPLITUDE_SOLVER_HH
