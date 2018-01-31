#ifndef GENERATOR_ACF_GENERATOR_HH
#define GENERATOR_ACF_GENERATOR_HH

#include <iosfwd>

#include "discrete_function.hh"
#include "domain.hh"
#include "plain_wave_profile.hh"

namespace arma {

	namespace generator {

		template <class T>
		class ACF_generator {

		public:
			typedef Discrete_function<T,3> array_type;
			typedef Domain<T,3> domain_type;

		private:
			typedef std::pair<Array3D<T>,domain_type> array_and_domain;

		private:
			/// Plain wave profile analytic approximation.
			Plain_wave_profile _func = Plain_wave_profile::Cosine;
			/// Wave amplitude.
			T _amplitude = T(1);
			/// Angular velocity.
			T _velocity = T(1);
			/// Angular wave numbers (x,y).
			Vec2D<T> _wavenum = Vec2D<T>(0.8, 0.0);
			/// Exponential decay factor (t,x,y).
			Vec3D<T> _alpha = Vec3D<T>(0.06, 0.06, 0.06);
			/// The number of waves in the wave group. Might be a fraction.
			T _nwaves = 1.5;
			/// Maximum variance difference when finding optimal
			/// wavy surface size.
			T _varepsilon = T(1e-3);
			T _chopepsilon = T(1e-10);

		public:

			ACF_generator() = default;

			~ACF_generator() = default;

			ACF_generator(const ACF_generator&) = default;

			ACF_generator(ACF_generator&&) = default;

			ACF_generator&
			operator=(const ACF_generator&) = default;

			ACF_generator&
			operator=(ACF_generator&&) = default;

		public:

			inline array_type
			operator()() {
				return this->generate();
			}

			array_type
			generate();

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, ACF_generator<X>& rhs);

		private:

			/**
			   \brief Generate wavy surface of optimal size.

			   First, generate 2x2x2 wavy surface. Then, gradually double
			   the size until the variance change in subsequent
			   iterations is lesser than epsilon.
			 */
			array_and_domain
			generate_optimal_wavy_surface();

			Array3D<T>
			add_exponential_decay(
				Array3D<T> surface,
				const domain_type& domain
			);

		};

		template <class T>
		std::istream&
		operator>>(std::istream& in, ACF_generator<T>& rhs);

	}

}

#endif // vim:filetype=cpp
