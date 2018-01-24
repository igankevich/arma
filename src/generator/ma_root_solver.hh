#ifndef GENERATOR_MA_ROOT_SOLVER_HH
#define GENERATOR_MA_ROOT_SOLVER_HH

#include "types.hh"

#include <gsl/gsl_multiroots.h>

namespace arma {

	namespace generator {

		/**
		   \brief
		   Multi-dimensional solver that determines MA model coefficients from
		      ACF.
		   \date 2018-01-24
		   \author Ivan Gankevich
		 */
		template <class T>
		class MA_root_solver {

		private:
			Array3D<T> _acf;
			bool _usederivatives = false;
			union {
				gsl_multiroot_fsolver* _fsolver;
				gsl_multiroot_fdfsolver* _fdfsolver;
			};
			int _maxiterations = 1000;
			T _maxresidual = T(1e-5);

		public:

			explicit
			MA_root_solver(Array3D<T> acf, bool use_derivatives=false);

			~MA_root_solver();

			MA_root_solver(const MA_root_solver&) = delete;

			MA_root_solver(MA_root_solver&&) = delete;

			MA_root_solver&
			operator=(const MA_root_solver&) = delete;

			MA_root_solver&
			operator=(MA_root_solver&&) = delete;

			Array3D<T>
			solve();

			inline Array3D<T>
			operator()() {
				return this->solve();
			}

			inline int
			max_iterations() const noexcept {
				return this->_maxiterations;
			}

			inline void
			max_iterations(int rhs) noexcept {
				this->_maxiterations = rhs;
			}

			inline T
			max_residual() const noexcept {
				return this->_maxresidual;
			}

			inline void
			max_residual(T rhs) noexcept {
				this->_maxresidual = rhs;
			}

		private:

			Array3D<T>
			solve_with_derivatives();

			Array3D<T>
			solve_without_derivatives();

		};

	}

}

#endif // vim:filetype=cpp
