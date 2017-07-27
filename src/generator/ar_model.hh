#ifndef AR_MODEL_HH
#define AR_MODEL_HH

#include "types.hh"
#include "arma.hh"
#include "basic_arma_model.hh"
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
		struct AR_model: public Basic_ARMA_model<T> {

			typedef Discrete_function<T,3> acf_type;

			AR_model() = default;

			inline explicit
			AR_model(acf_type acf, Shape3D order):
			Basic_ARMA_model<T>(acf, order),
			_phi(order)
			{}

			inline Array3D<T>
			coefficients() const {
				return _phi;
			}

			inline T
			white_noise_variance() const override {
				return white_noise_variance(_phi);
			}

			void validate() const override;

			/**
			Generate wavy surface realisation.
			*/
			void
			operator()(
				Array3D<T>& zeta,
				Array3D<T>& eps,
				const Domain3D& subdomain
			);

			inline void
			determine_coefficients() override {
				// determine_coefficients_iteratively();
				determine_coefficients_old(_doleastsquares);
			}

		protected:
			T
			white_noise_variance(Array3D<T> phi) const;

			void write(std::ostream& out) const override;
			void read(std::istream& in) override;

		private:

			void
			determine_coefficients_old(bool do_least_squares);

			/**
			Darbin algorithm. Partial autocovariation function \f$\phi_{k,j}\f$,
			where k --- AR process order, j --- coefficient index.
			*/
			void
			determine_coefficients_iteratively();

			Array3D<T> _phi;
			bool _doleastsquares = false;
		};

	}

}

#endif // AR_MODEL_HH
