#ifndef GENERATOR_ACF_GENERATOR_HH
#define GENERATOR_ACF_GENERATOR_HH

#include <cmath>
#include <iosfwd>

#include "discrete_function.hh"
#include "domain.hh"
#include "physical_constants.hh"
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
			Vec3D<T> _nwaves = Vec3D<T>(1.5, 1.5, 1.5);
			/// Maximum variance difference when finding optimal
			/// wavy surface size.
			T _varepsilon = T(1e-3);
			T _chopepsilon = T(1e-10);
			Shape3D _acfshape = Shape3D(16,16,16);

		public:

			ACF_generator() = default;

			~ACF_generator() = default;

			ACF_generator(const ACF_generator&) = default;

			ACF_generator(ACF_generator&&) = default;

			ACF_generator&
			operator=(const ACF_generator&) = default;

			ACF_generator&
			operator=(ACF_generator&&) = default;

			inline array_type
			operator()() {
				return this->generate();
			}

			array_type
			generate();

			inline T
			amplitude() const noexcept {
				return this->_amplitude;
			}

			inline T
			wave_height() const noexcept {
				return T(2)*this->_amplitude;
			}

			inline T
			wave_height_x() const noexcept {
				return this->has_x() ?  this->wave_height() : T(1) / T(0);
			}

			inline T
			wave_height_y() const noexcept {
				return this->has_y() ?  this->wave_height() : T(1) / T(0);
			}

			inline T
			wave_length_x() const noexcept {
				using constants::_2pi;
				using std::abs;
				return _2pi<T> / abs(this->_wavenum(0));
			}

			inline T
			wave_length_y() const noexcept {
				using constants::_2pi;
				using std::abs;
				return _2pi<T> / abs(this->_wavenum(1));
			}

			inline T
			wave_period() const noexcept {
				using constants::_2pi;
				using std::abs;
				return _2pi<T> / abs(this->_velocity);
			}

			inline bool
			has_x() const {
				return std::isfinite(this->wave_length_x());
			}

			inline bool
			has_y() const {
				return std::isfinite(this->wave_length_y());
			}

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, ACF_generator<X>& rhs);

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const ACF_generator<X>& rhs);

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

			inline Grid<T,3>
			acf_grid(const Shape3D& shape) const noexcept {
//				const Vec3D<T> k(
//					this->_velocity,
//					this->_wavenum(0),
//					this->_wavenum(1)
//				);
				return Grid<T,3>{shape, this->_nwaves};
			}

			inline Domain<T,3>
			acf_domain(const Shape3D& wave_shape) const noexcept {
				Vec3D<T> r = this->_nwaves;
//				const Vec3D<T> k(
//					this->_velocity,
//					this->_wavenum(0),
//					this->_wavenum(1)
//				);
//				r /= k;
//				r = blitz::where(blitz::isfinite(r), r, this->_nwaves);
				return Domain<T,3>{-r, r, wave_shape+1};
			}

		};

		template <class T>
		std::istream&
		operator>>(std::istream& in, ACF_generator<T>& rhs);

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const ACF_generator<T>& rhs);

	}

}

#endif // vim:filetype=cpp
