#ifndef GENERATOR_BASIC_ARMA_MODEL_HH
#define GENERATOR_BASIC_ARMA_MODEL_HH

#include "basic_model.hh"
#include "discrete_function.hh"
#include "nonlinear/nit_transform.hh"
#include "params.hh"
#include <chrono>

namespace arma {

	namespace generator {

		template <class T>
		class Basic_ARMA_model: public virtual Basic_model<T> {

		public:
			typedef std::chrono::high_resolution_clock clock_type;
			typedef Discrete_function<T,3> acf_type;
			typedef nonlinear::NIT_transform<T> transform_type;

		protected:
			/// Autocovariate function (ACF) of the process.
			acf_type _acf;
			/// Process order.
			Shape3D _order = Shape3D(0,0,0);
			/// Whether seed PRNG or not. This flag is needed for
			/// reproducible tests.
			bool _noseed = false;
			transform_type _nittransform;
			bool _linear = true;

			inline clock_type::rep
			newseed() noexcept {
				#if defined(ARMA_NO_PRNG_SEED)
				return clock_type::rep(0);
				#else
				return
					this->_noseed
					? clock_type::rep(0)
					: clock_type::now().time_since_epoch().count();
				#endif
			}

			virtual Array3D<T> do_generate() = 0;

		public:
			inline
			Basic_ARMA_model() = default;

			inline explicit
			Basic_ARMA_model(acf_type acf, Shape3D order):
			_acf(acf), _order(order)
			{}

			const Shape3D&
			order() const noexcept {
				return this->_order;
			}

			inline const acf_type&
			acf() const noexcept{
				return this->_acf;
			}

			inline T
			acf_variance() const noexcept {
				return this->_acf(0,0,0);
			}

			inline void
			setacf(const acf_type& rhs) {
				this->_acf.reference(rhs);
			}

			virtual T white_noise_variance() const = 0;
			virtual void determine_coefficients() = 0;
			void verify(Array3D<T> zeta) const override;

			virtual sys::parameter_map::map_type
			parameters();

			virtual void
			generate_surface(
				Array3D<T>& zeta,
				Array3D<T>& eps,
				const Domain3D& subdomain
			) = 0;

		public:
			Array3D<T> generate() override;

		};

	}

}

#endif // GENERATOR_BASIC_ARMA_MODEL_HH
