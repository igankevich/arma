#ifndef MA_MODEL_HH
#define MA_MODEL_HH

#include "types.hh"
#include "ma_algorithm.hh"
#include "arma.hh"
#include "basic_arma_model.hh"
#include "discrete_function.hh"

namespace arma {

	namespace generator {

		/**
		\brief Uses moving average process, propagating waves.
		\ingroup generators
		*/
		template <class T>
		struct MA_model: public Basic_ARMA_model<T> {

			typedef Discrete_function<T,3> acf_type;

			MA_model() = default;

			inline explicit
			MA_model(acf_type acf, Shape3D order):
			Basic_ARMA_model<T>(acf, order),
			_theta(order)
			{}

			inline Array3D<T>
			coefficients() const {
				return this->_theta;
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
				return this->_varwn;
			}

			void validate() const override;

			void
			determine_coefficients() override;

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const MA_model<X>& rhs);

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, MA_model<X>& rhs);

		protected:
			void
			generate_surface(
				Array3D<T>& zeta,
				Array3D<T>& eps,
				const Domain3D& subdomain
			);

			T
			white_noise_variance(const Array3D<T>& theta) const;

			Array3D<T> do_generate() override;

			void write(std::ostream& out) const override;
			void read(std::istream& in) override;

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
			Array3D<T> _theta;
			MA_algorithm _algo = MA_algorithm::Fixed_point_iteration;
			int _maxiter = 1000;
			T _eps = T(1e-5);
			T _minvarwn = T(1e-6);
			/// White noise variance.
			T _varwn = T(0);
		};

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const MA_model<T>& rhs) {
			rhs.MA_model<T>::write(out);
			return out;
		}

		template <class T>
		std::istream&
		operator>>(std::istream& in, MA_model<T>& rhs) {
			rhs.MA_model<T>::read(in);
			return in;
		}


	}

}

#endif // MA_MODEL_HH
