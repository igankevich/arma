#ifndef GENERATOR_LONGUET_HIGGINS_HH
#define GENERATOR_LONGUET_HIGGINS_HH

#include <istream>
#include <ostream>

#include "types.hh"
#include "discrete_function.hh"
#include "domain.hh"
#include "grid.hh"

namespace arma {

	namespace generator {

		/**
		\brief Uses Longuet---Higgins model, small-amplitude waves.
		*/
		template <class T>
		class Longuet_Higgins_model {
			Domain<T,2> _spec_domain;
			Vec2D<int> _spec_subdomain;
			T _waveheight;
			Grid<T,3> _outgrid;
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

			void
			setgrid(const Grid<T,3>& rhs) noexcept {
				_outgrid = rhs;
			}

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, Longuet_Higgins_model<X>& rhs);

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const Longuet_Higgins_model<X>& rhs);

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

		template <class T>
		std::istream&
		operator>>(std::istream& in, Longuet_Higgins_model<T>& rhs);

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const Longuet_Higgins_model<T>& rhs);

	}

}

#endif // GENERATOR_LONGUET_HIGGINS_HH
