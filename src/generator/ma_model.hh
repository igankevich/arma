#ifndef MA_MODEL_HH
#define MA_MODEL_HH

#include "arma.hh"
#include "basic_arma_model.hh"
#include "discrete_function.hh"
#include "ma_algorithm.hh"
#include "types.hh"

namespace arma {

	namespace generator {

		/**
		   \brief Uses moving average process, propagating waves.
		   \ingroup generators
		 */
		template <class T>
		class MA_model: public Basic_ARMA_model<T> {

		public:
			typedef Discrete_function<T,3> acf_type;

		private:
			/// MA model coefficients.
			Array3D<T> _theta;
			/// The algorithm that determines MA model coefficients.
			MA_algorithm _algo = MA_algorithm::Fixed_point_iteration;

		public:

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

			void
			validate() const override;

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

			Array3D<T>
			do_generate() override;

			void
			write(std::ostream& out) const override;

			void
			read(std::istream& in) override;

		private:

			/**
			   Solve nonlinear system with fixed-point iteration algorithm to
			      find
			   moving-average coefficients \f$\theta\f$.
			 */
			void
			fixed_point_iteration();

			void
			recompute_acf(Array3D<T> acf_orig, Array3D<T> phi);

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
