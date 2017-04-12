#ifndef MA_MODEL_HH
#define MA_MODEL_HH

#include "types.hh"
#include "ma_algorithm.hh"
#include "arma.hh"
#include "model.hh"

namespace arma {

	template <class T>
	struct Moving_average_model: public virtual Basic_ARMA_model<T> {

		Moving_average_model() = default;

		inline explicit
		Moving_average_model(Array3D<T> acf, Shape3D order):
		_acf(acf),
		_theta(order)
		{}

		inline Array3D<T>
		acf() const {
			return _acf;
		}

		inline void
		setacf(Array3D<T> acf) {
			_acf.resize(acf.shape());
			_acf = acf;
		}

		inline T
		acf_variance() const {
			return _acf(0, 0, 0);
		}

		inline Array3D<T>
		coefficients() const {
			return _theta;
		}

		inline const Shape3D&
		order() const {
			return _theta.shape();
		}

		/**
		Compute white noise variance via the following formula.
		\f[
		    \sigma_\alpha^2 = \frac{\gamma_0}{
		        1
		        +
		        \sum\limits_{i=0}^{n_1}
		        \sum\limits_{i=0}^{n_2}
		        \sum\limits_{k=0}^{n_3}
		        \theta_{i,j,k}^2
		    }
		\f]
		*/
		inline T
		white_noise_variance() const override {
			return white_noise_variance(_theta);
		}

		inline void
		validate() const override {
			validate_process(_theta);
		}

		void
		operator()(Array3D<T>& zeta, Array3D<T>& eps) override;

		template <class X>
		friend std::istream&
		operator>>(std::istream& in, Moving_average_model<X>& rhs);

		template <class X>
		friend std::ostream&
		operator<<(std::ostream& out, const Moving_average_model<X>& rhs);

		void
		determine_coefficients() override;

	protected:
		T
		white_noise_variance(const Array3D<T>& theta) const;

	private:
		/**
		Solve nonlinear system with fixed-point iteration algorithm to find
		moving-average coefficients \f$\theta\f$.

		\param max_iterations Maximal no. of iterations.
		\param eps
		\parblock
		    Maximal difference between values of white
		    noise variance in successive iterations.
		\endparblock
		\param min_var_wn
		\parblock
		    Maximal value of white noise variance considered to be
		    nought.
		\endparblock
		*/
		void
		fixed_point_iteration(int max_iterations, T eps, T min_var_wn);

		void
		newton_raphson(int max_iterations, T eps, T min_var_wn);

		void
		recompute_acf(Array3D<T> acf_orig, Array3D<T> phi);

	private:
		Array3D<T> _acf;
		Array3D<T> _theta;
		MA_algorithm _algo = MA_algorithm::Fixed_point_iteration;
		int _maxiter = 1000;
		T _eps = T(1e-5);
		T _minvarwn = T(1e-6);
	};


	template <class T>
	std::istream&
	operator>>(std::istream& in, Moving_average_model<T>& rhs);

	template <class T>
	std::ostream&
	operator<<(std::ostream& out, const Moving_average_model<T>& rhs);
}

#endif // MA_MODEL_HH
