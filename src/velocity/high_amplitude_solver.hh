#ifndef VELOCITY_HIGH_AMPLITUDE_SOLVER_HH
#define VELOCITY_HIGH_AMPLITUDE_SOLVER_HH

#include "types.hh"
#include "linear_solver.hh"

namespace arma {

	namespace velocity {

		/**
		\brief Uses analytic formula based on Fourier transforms,
		arbitrary amplitude waves.
		\ingroup solvers
		*/
		template <class T>
		class High_amplitude_solver: public Linear_solver<T> {

		protected:
			/**
			Compute multiplier function as
			\f[
				F(x, y) = \frac{ \zeta_t }{
					(\zeta_x + \zeta_y) /
					\sqrt{\smash[b]{1 + \zeta_x^2 + \zeta_y^2}}
					- \zeta_x - \zeta_y - 1
				}
			\f]
			instead of \f$F(x,y)=\zeta_t\f$ as in linear case.
			*/
			void
			precompute(const Discrete_function<T,3>& zeta, const int idx_t) override;

			void
			precompute(const Discrete_function<T,3>& zeta) override;

		};

	}
}

#endif // VELOCITY_HIGH_AMPLITUDE_SOLVER_HH
