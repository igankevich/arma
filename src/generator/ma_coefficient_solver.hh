#ifndef GENERATOR_MA_COEFFICIENT_SOLVER_HH
#define GENERATOR_MA_COEFFICIENT_SOLVER_HH

#include "arma.hh"

namespace arma {

	namespace generator {

		/**
		   \brief
		   Multi-dimensional solver that determines MA model coefficients from
		      ACF.
		   \date 2018-01-24
		   \author Ivan Gankevich

		   The solver uses fixed point iteration to find the coefficients.
		 */
		template <class T>
		class MA_coefficient_solver {

		private:
			/// Auto-covariance function.
			Array3D<T> _acf;
			/// MA model order.
			Shape3D _order;
			/// Maximum number of iterations.
			int _maxiterations = 1000;
			T _maxresidual = T(1e-5);
			/**
			\brief Minimal white noise variance.

			White noise variance, that is lower than this value,
			is considered to be nought.
			*/
			T _minvarwn = T(1e-6);
			/// Minimal white noise variance difference between iterations.
			T _epsvarwn = T(1e-5);

		public:

			MA_coefficient_solver() = default;

			explicit
			MA_coefficient_solver(Array3D<T> acf, const Shape3D& order);

			~MA_coefficient_solver() = default;

			MA_coefficient_solver(const MA_coefficient_solver&) = delete;

			MA_coefficient_solver(MA_coefficient_solver&&) = delete;

			MA_coefficient_solver&
			operator=(const MA_coefficient_solver&) = delete;

			MA_coefficient_solver&
			operator=(MA_coefficient_solver&&) = delete;

			Array3D<T>
			solve();

			inline Array3D<T>
			operator()() {
				return this->solve();
			}

			inline const Shape3D&
			order() const noexcept {
				return this->_order;
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

			inline void
			min_white_noise_variance(T rhs) noexcept {
				this->_minvarwn = rhs;
			}

			inline T
			min_white_noise_variance() const noexcept {
				return this->_minvarwn;
			}

			inline void
			min_white_noise_variance_delta(T rhs) noexcept {
				this->_epsvarwn = rhs;
			}

			inline T
			min_white_noise_variance_delta() const noexcept {
				return this->_epsvarwn;
			}

			inline T
			white_noise_variance(const Array3D<T>& theta) const {
				return MA_white_noise_variance(this->_acf, theta);
			}

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, MA_coefficient_solver<X>& rhs);

		private:

			Array3D<T>
			solve_fixed_point_iteration();

			Array3D<T>
			solve_non_convex();

			Array3D<T>
			solve_bisection();

		};

		template <class T>
		std::istream&
		operator>>(std::istream& in, MA_coefficient_solver<T>& rhs);

	}

}

#endif // vim:filetype=cpp
