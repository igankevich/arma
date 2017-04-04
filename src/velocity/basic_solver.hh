#ifndef VELOCITY_BASIC_SOLVER_HH
#define VELOCITY_BASIC_SOLVER_HH

#if ARMA_OPENMP
#include <omp.h>
#endif
#include "types.hh"
#include "domain.hh"
#include "discrete_function.hh"

namespace arma {

	namespace velocity {

		template<class T>
		class Velocity_potential_solver {

		protected:
			Vec2D<T> _wnmax;
			T _depth;
			Domain2<T> _domain;

			virtual void
			precompute(const Discrete_function<T,3>& zeta) {}

			virtual void
			precompute(const Discrete_function<T,3>& zeta, const int idx_t) {}

			virtual Array2D<T>
			compute_velocity_field_2d(
				const Discrete_function<T,3>& zeta,
				const Shape2D arr_size,
				const T z,
				const int idx_t
			) { return Array2D<T>(); };

			virtual void
			write(std::ostream& out) const;

			virtual void
			read(std::istream& in);

		public:
			Velocity_potential_solver() = default;
			Velocity_potential_solver(const Velocity_potential_solver&) = default;
			Velocity_potential_solver(Velocity_potential_solver&&) = default;
			virtual ~Velocity_potential_solver() = default;

			/**
			\param[in] zeta      ocean wavy surface
			\param[in] subdomain region of zeta
			\param[in] z         a coordinate \f$z\f$ in which to compute velocity
								 potential
			\param[in] idx_t     a time point in which to compute velocity potential,
								 specified as index of zeta
			*/
			virtual Array4D<T>
			operator()(const Discrete_function<T,3>& zeta);

			inline const Domain2<T>
			domain() const noexcept {
				return _domain;
			}

			inline friend std::ostream&
			operator<<(std::ostream& out, const Velocity_potential_solver& rhs) {
				rhs.write(out);
				return out;
			}

			inline friend std::istream&
			operator>>(std::istream& in, Velocity_potential_solver& rhs) {
				rhs.read(in);
				return in;
			}

		};

	}
}

#endif // VELOCITY_BASIC_SOLVER_HH
