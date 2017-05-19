#ifndef VELOCITY_PLAIN_WAVE_SOLVER_HH
#define VELOCITY_PLAIN_WAVE_SOLVER_HH

#include "basic_solver.hh"
#include "generator/plain_wave.hh"
#include "types.hh"

namespace arma {

	namespace velocity {

		/**
		\brief Uses analytic formulae from linear wave theory, sine and cosine waves.
		*/
		template <class T>
		class Plain_wave_solver: public Velocity_potential_solver<T> {

			typedef Plain_wave<T> wave_type;
			wave_type _waves;

		protected:
			Array2D<T>
			compute_velocity_field_2d(
				const Discrete_function<T,3>& zeta,
				const Shape2D arr_size,
				const T z,
				const int idx_t
			) override;

			void
			write(std::ostream& out) const override;

			void
			read(std::istream& in) override;

		};
	}
}

#endif // VELOCITY_PLAIN_WAVE_SOLVER_HH
