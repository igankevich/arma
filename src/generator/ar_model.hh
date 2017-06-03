#ifndef AR_MODEL_HH
#define AR_MODEL_HH

#include "types.hh"
#include "arma.hh"
#include "model.hh"
#include "discrete_function.hh"

/// @file
/// File with subroutines for AR model, Yule-Walker equations
/// and some others.

namespace arma {

	namespace generator {

		/**
		\brief Uses autoregressive process, standing waves.
		*/
		template <class T>
		struct AR_model: public virtual Basic_ARMA_model<T> {

			typedef Discrete_function<T,3> acf_type;

			AR_model() = default;

			inline explicit
			AR_model(acf_type acf, Shape3D order):
			_acf(acf), _phi(order)
			{}

			inline acf_type
			acf() const {
				return _acf;
			}

			inline void
			setacf(acf_type acf) {
				_acf.resize(acf.shape());
				_acf = acf;
			}

			inline T
			acf_variance() const {
				return _acf(0, 0, 0);
			}

			inline Array3D<T>
			coefficients() const {
				return _phi;
			}

			inline const Shape3D&
			order() const {
				return _phi.shape();
			}

			inline T
			white_noise_variance() const override {
				return white_noise_variance(_phi);
			}

			inline void
			validate() const override {
				validate_process(_phi);
			}

			/**
			Generate wavy surface realisation.
			*/
			void
			operator()(
				Array3D<T>& zeta,
				Array3D<T>& eps,
				const Domain3D& subdomain
			) override;

			inline void
			determine_coefficients() override {
				// determine_coefficients_iteratively();
				determine_coefficients_old(_doleastsquares);
			}

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, AR_model<X>& rhs);

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const AR_model<X>& rhs);

		protected:
			T
			white_noise_variance(Array3D<T> phi) const;

		private:

			void
			determine_coefficients_old(bool do_least_squares);

			/**
			Darbin algorithm. Partial autocovariation function \f$\phi_{k,j}\f$,
			where k --- AR process order, j --- coefficient index.
			*/
			void
			determine_coefficients_iteratively();

			acf_type _acf;
			Array3D<T> _phi;
			bool _doleastsquares = false;
		};

		template <class T>
		std::istream&
		operator>>(std::istream& in, AR_model<T>& rhs);

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const AR_model<T>& rhs);

	}

}

#endif // AR_MODEL_HH
