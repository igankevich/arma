#ifndef VELOCITY_BASIC_SOLVER_HH
#define VELOCITY_BASIC_SOLVER_HH

#if ARMA_OPENMP
#include <omp.h>
#endif
#include "types.hh"
#include "domain.hh"
#include "discrete_function.hh"

namespace arma {

	/**
	\brief Velocity potential field solvers.
	*/
	namespace velocity {

		/**
		\defgroup solvers Velocity potential solvers
		\brief Solvers which determine velocity potential field under wavy surface.
		*/

		/**
		\brief Base class for all velocity potential field solvers.
		\ingroup solvers
		*/
		template<class T>
		class Velocity_potential_solver {

		protected:
			typedef Domain2<T> domain2_type;

		protected:
			/// Wave number range in \f$X\f$ and \f$Y\f$ dimensions.
			domain2_type _wnmax;
			/// Water depth.
			T _depth = 0;
			domain2_type _domain;

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

			Velocity_potential_solver():
			_wnmax(),
			_domain()
			{}

			Velocity_potential_solver(const Velocity_potential_solver&) = default;
			Velocity_potential_solver(Velocity_potential_solver&&) = default;
			virtual ~Velocity_potential_solver() = default;

			/**
			\param[in] zeta ocean wavy surface
			\return velocity potential field with \f$(t,z,x,y)\f$ dimensions
			*/
			virtual Array4D<T>
			operator()(const Discrete_function<T,3>& zeta);

			inline const Domain2<T>
			domain() const noexcept {
				return this->_domain;
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
