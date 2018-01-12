#ifndef YULE_WALKER_HH
#define YULE_WALKER_HH

#include <cmath>

#include "types.hh"

namespace arma {

	/**
	   \brief
	   Computes AR model coefficients using an order-recursvive method
	   from \cite Choi1999.
	   \date 2018-01-12
	   \author Ivan Gankevich

	   The AR model order is determined automatically. The algorithm works for
	   square ACFs (i.e. ACF with the same number of points along each
	   dimension).
	 */
	template <class T>
	class Yule_walker_solver {

	public:
		/// Array element type.
		typedef T value_type;
		/// Three-dimensional array type.
		typedef Array3D<T> array_type;

	private:
		/**
		   \brief
		   Three-dimensional auto-correlation function \f$\rho\f$
		   with \f$\rho_{0,0,0}=1\f$.
		 */
		array_type _acf;
		/// Variance of the ACF.
		T _variance = 0;
		/// Maximum AR model order.
		int _maxorder = 0;
		/**
		   \brief
		   Maximum variance difference between subsequent algorithm iterations.
		   The algorithm stops when \f$\left|\sigma_{i+1}^2-\sigma_{i}^2\right|
		   \leq\epsilon\f$.
		 */
		T _varepsilon = T(1e-5);
		/**
		   \brief
		   Determine the order of AR model automatically by comparing
		   the variance in subsequent algorithm iterations.
		 */
		bool _determinetheorder = true;
		/**
		   \brief
		   Maximum ACF value. Smaller values are considered noughts and are
		   removed from the end of ACF.
		 */
		T _chopepsilon = T(1e-10);
		/**
		   \brief
		   Determines whether to chop the result removing adjacent values
		   smaller than \f$\epsilon_{\text{chop}}\f$ from the end of the
		   resulting ACF.
		 */
		bool _chop = true;

	public:

		/**
		   \brief
		   Constructs a solver with specified auto-correlation function and
		   variance.

		   \param[in] acf three-dimensional auto-correlation function \f$\rho\f$
		              with \f$\rho_{0,0,0}=1\f$
		   \param[in] variance desired \f$\sigma^2\f$
		 */
		Yule_walker_solver(array_type acf, const T variance);

		~Yule_walker_solver() = default;

		Yule_walker_solver(const Yule_walker_solver&) = delete;

		Yule_walker_solver&
		operator=(const Yule_walker_solver&) = delete;

		/**
		\brief Solve Yule---Walker system of equations.
		\return three-dimensional array of coefficients
		*/
		array_type
		solve();

		/// \copydoc solve
		inline array_type
		operator()() {
			return this->solve();
		}

		void
		max_order(int rhs);

		/// \copydoc _maxorder
		inline int
		max_order() const noexcept {
			return this->_maxorder;
		}

		/// \copydoc _determinetheorder
		inline bool
		determine_the_order() const noexcept {
			return this->_determinetheorder;
		}

		inline void
		determine_the_order(bool rhs) noexcept {
			this->_determinetheorder = rhs;
		}

		inline void
		do_not_determine_the_order() noexcept {
			this->_determinetheorder = false;
		}

		inline void
		chop(bool rhs) noexcept {
			this->_chop = rhs;
		}

		/// \copydoc _chop
		inline bool
		chop() const noexcept {
			return this->_chop;
		}

		inline void
		do_not_chop() noexcept {
			this->_chop = false;
		}

		/// \copydoc _chopepsilon
		inline value_type
		chop_epsilon() const noexcept {
			return this->_chopepsilon;
		}

		inline void
		chop_epsilon(value_type rhs) noexcept {
			this->_chopepsilon = rhs;
		}

		/// \copydoc _varepsilon
		inline value_type
		var_epsilon() const noexcept {
			return this->_varepsilon;
		}

		inline void
		var_epsilon(value_type rhs) noexcept {
			this->_varepsilon = rhs;
		}

	private:

		inline bool
		variance_has_not_changed_much(T var0, T var1) noexcept {
			return this->_determinetheorder &&
			       std::abs(var1-var0) < this->_varepsilon;
		}

	};

}

#endif // vim:filetype=cpp
