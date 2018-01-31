#ifndef GENERATOR_PLAIN_WAVE_HH
#define GENERATOR_PLAIN_WAVE_HH

#include <blitz/array.h>
#include <cmath>

#include "physical_constants.hh"

namespace arma {

	namespace generator {

		/*
		template <class T>
		class Plain_wave: public blitz::TinyVector<T,5> {

		public:

			Plain_wave() = default;

			~Plain_wave() = default;

			Plain_wave(const Plain_wave&) = default;

			Plain_wave&
			operator=(const Plain_wave&) = default;

			inline T
			amplitude() const noexcept {
				return this->operator()(0);
			}

			inline T
			wavenum_x() const noexcept {
				return this->operator()(1);
			}

			inline T
			wavenum_y() const noexcept {
				return this->operator()(2);
			}

			inline T
			velocity() const noexcept {
				return this->operator()(3);
			}

			inline T
			phase() const noexcept {
				return this->operator()(4);
			}

		};
		*/

		template <class T>
		inline T
		sine_wave(
			const T amplitude,
			const T kx,
			const T ky,
			const T velocity,
			const T phase,
			const T x,
			const T y,
			const T t
		) {
			using arma::constants::_2pi;
			using std::sin;
			return amplitude * sin(_2pi<T>*(kx*x + ky*y - velocity*t) + phase);
		}

		template <class T>
		inline T
		cosine_wave(
			const T amplitude,
			const T kx,
			const T ky,
			const T velocity,
			const T phase,
			const T x,
			const T y,
			const T t
		) {
			using arma::constants::pi_div_2;
			return sine_wave(
				amplitude,
				kx, ky,
				velocity,
				phase + pi_div_2<T>,
				x, y, t
			);
		}

		template <class T>
		inline T
		stokes_wave(
			const T amplitude,
			const T kx,
			const T ky,
			const T velocity,
			const T phase,
			const T x,
			const T y,
			const T t
		) {
			using arma::constants::_2pi;
			using std::cos;
			using std::sqrt;
			const T theta = _2pi<T>*(kx*x + ky*y - velocity*t) + phase;
			const T steepness = _2pi<T> * amplitude * sqrt(kx*kx + ky*ky);
			return amplitude * (
				cos(theta)
				+ T(0.5)*steepness*cos(T(2)*theta)
				+ (T(3)/T(8))*steepness*steepness*cos(T(3)*theta)
			);
		}

		template <class T>
		inline T
		standing_wave(
			const T amplitude,
			const T kx,
			const T ky,
			const T velocity,
			const T phase,
			const T x,
			const T y,
			const T t
		) {
			using arma::constants::_2pi;
			using std::cos;
			return amplitude *
				cos(_2pi<T>*(kx*x + ky*y) + phase) *
				cos(_2pi<T>*(velocity*t));
		}

	}

}

#endif // vim:filetype=cpp
