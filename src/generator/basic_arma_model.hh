#ifndef GENERATOR_BASIC_ARMA_MODEL_HH
#define GENERATOR_BASIC_ARMA_MODEL_HH

#include <vector>

#include "basic_model.hh"
#include "discrete_function.hh"
#include "nonlinear/nit_transform.hh"
#include "params.hh"

namespace arma {

	namespace generator {

		/// Base class for AR, MA and ARMA models.
		template <class T>
		class Basic_ARMA_model:
			public virtual Basic_model<T>{

		public:
			typedef Discrete_function<T,3> acf_type;
			typedef nonlinear::NIT_transform<T> transform_type;

		protected:
			/// Autocovariate function (ACF) of the process.
			acf_type _acf;
			/// Process order.
			Shape3D _order = Shape3D(0,0,0);
			transform_type _nittransform;
			bool _linear = true;

			virtual Array3D<T>
			do_generate() = 0;

			Array3D<T>
			generate_white_noise();

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
			acf() const noexcept {
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

			#if ARMA_BSCHEDULER
			void
			act() override;

			void
			react(bsc::kernel* child) override;

			void
			write(sys::pstream& out) const override;

			void
			read(sys::pstream& in) override;

			#endif

			virtual T
			white_noise_variance() const = 0;

			virtual void
			determine_coefficients() = 0;

			void
			verify(Array3D<T> zeta) const override;

			virtual sys::parameter_map::map_type
			parameters();

		public:
			Array3D<T>
			generate() override;

		};

	}

}

#endif // GENERATOR_BASIC_ARMA_MODEL_HH
