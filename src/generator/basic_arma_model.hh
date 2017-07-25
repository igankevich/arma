#ifndef GENERATOR_BASIC_ARMA_MODEL_HH
#define GENERATOR_BASIC_ARMA_MODEL_HH

#include "basic_model.hh"
#include <chrono>
#include "discrete_function.hh"
#include "nonlinear/nit_transform.hh"

namespace arma {

	namespace generator {

		template <class T>
		class Basic_ARMA_model: public virtual Basic_model<T> {

		protected:
			typedef std::chrono::high_resolution_clock clock_type;
			typedef Discrete_function<T,3> acf_type;
			acf_type _acf;
			/// The size of partitions that are computed in parallel.
			Shape3D _partition;
			/// Whether seed PRNG or not. This flag is needed for
			/// reproducible tests.
			bool _noseed = false;
			nonlinear::NIT_transform<T> _nittransform;
			bool _linear = true;

			Shape3D
			get_partition_shape(Shape3D order, int nprngs);

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

			virtual Shape3D order() const = 0;

			inline acf_type
			acf() const noexcept{
				return this->_acf;
			}

		public:
			Array3D<T> generate() override;

		private:
			Array3D<T> do_generate();

		};

	}

}

#endif // GENERATOR_BASIC_ARMA_MODEL_HH
