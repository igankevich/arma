#ifndef GENERATOR_LONGUET_HIGGINS_HH
#define GENERATOR_LONGUET_HIGGINS_HH

#include <istream>
#include <ostream>

#include "types.hh"
#include "discrete_function.hh"
#include "domain.hh"
#include "grid.hh"
#include "basic_model.hh"

namespace arma {

	namespace generator {

		/**
		\brief Uses Longuet---Higgins model, small-amplitude waves.
		*/
		template <class T>
		class Longuet_Higgins_model: public Basic_model<T> {
			Domain<T,2> _spec_domain;
			Vec2D<int> _spec_subdomain;
			T _waveheight;
			Array2D<T> _coef;
			Array2D<T> _eps;

		public:
			inline void
			operator()(Discrete_function<T,3>& zeta) {
				operator()(zeta, zeta.domain());
			}

			inline void
			operator()(Discrete_function<T,3>& zeta, const Domain3D& subdomain) {
				generate(zeta, subdomain);
			}

			void
			determine_coefficients();

			void
			generate_white_noise();

			inline Array3D<T>
			generate() override {
				this->determine_coefficients();
				this->generate_white_noise();
				Discrete_function<T,3> zeta;
				zeta.resize(this->grid().num_points());
				zeta.setgrid(this->grid());
				operator()(zeta, zeta.domain());
				return zeta;
			}

			T white_noise_variance() const override {
				return T(0);
			}

			void validate() const override {}

			virtual void
			operator()(
				Array3D<T>& zeta,
				Array3D<T>& eps,
				const Domain3D& subdomain
			) override {
				throw std::runtime_error("not implemented");
			}

		protected:
			void write(std::ostream& out) const override;
			void read(std::istream& in) override;

		private:
			T
			approx_spectrum(T w, T theta, T h);

			Array2D<T>
			approximate_spectrum(const Domain<T,2>& domain, T wave_height);

			Array2D<T>
			determine_coefficients(const Domain<T,2>& sdom, T wave_height);

			void
			generate_surface(Discrete_function<T,3>& zeta, const Domain3D& subdomain);

			void
			generate(Discrete_function<T,3>& zeta, const Domain3D& subdomain);
		};

	}

}

#endif // GENERATOR_LONGUET_HIGGINS_HH
