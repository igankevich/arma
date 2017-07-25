#ifndef GENERATOR_BASIC_GENERATOR_HH
#define GENERATOR_BASIC_GENERATOR_HH

#include "types.hh"

namespace arma {

	// \brief Wavy surface generators.
	namespace generator {

		/// \brief A base class for wavy surface generators.
		template<class T>
		class Basic_generator {
		public:

			Basic_generator() = default;
			Basic_generator(const Basic_generator&) = default;
			Basic_generator(Basic_generator&&) = default;
			virtual ~Basic_generator() = default;

			virtual T white_noise_variance() const = 0;
			virtual void validate() const = 0;
			virtual void determine_coefficients() = 0;

			virtual void
			operator()(
				Array3D<T>& zeta,
				Array3D<T>& eps,
				const Domain3D& subdomain
			) = 0;

			inline void
			operator()(Array3D<T>& zeta, Array3D<T>& eps) {
				operator()(zeta, eps, zeta.domain());
			}

		};

	}

}

#endif // GENERATOR_BASIC_GENERATOR_HH
