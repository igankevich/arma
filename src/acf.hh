#ifndef ACF_HH
#define ACF_HH

#include <functional>
#include <unordered_map>
#include <string>
#include "types.hh"
#include "params.hh"
#include "grid.hh"
#include "validators.hh"
#include "discrete_function.hh"

/// @file
/// Mini-database of ACF approximations.

namespace arma {

	template <class T>
	Array3D<T>
	standing_wave_ACF(
		const Vec3D<T>& delta,
		const Shape3D& acf_size,
		T amplitude,
		T velocity,
		const Vec3D<T>& alpha,
		const Vec2D<T>& beta
	) {
		using blitz::exp;
		using blitz::cos;
		blitz::firstIndex i;
		blitz::secondIndex j;
		blitz::thirdIndex k;
		Array3D<T> acf(acf_size);
		acf = amplitude *
			exp(-(alpha(0)*i*delta(0) + alpha(1)*j*delta(1) + alpha(2)*k*delta(2))) *
			cos(velocity*i*delta(0)) *
			cos(beta(0)*j*delta(1)) *
			cos(beta(1)*k*delta(2));
		return acf;
	}

	template <class T>
	Array3D<T>
	propagating_wave_ACF(
		const Vec3D<T>& delta,
		const Shape3D& acf_size,
		T amplitude,
		T velocity,
		const Vec3D<T>& alpha,
		const Vec2D<T>& beta
	) {
		using blitz::exp;
		using blitz::cos;
		blitz::firstIndex i;
		blitz::secondIndex j;
		blitz::thirdIndex k;
		Array3D<T> acf(acf_size);
		acf = amplitude *
			exp(-(alpha(0)*i*delta(0) + alpha(1)*j*delta(1) + alpha(2)*k*delta(2))) *
			cos(velocity*i*delta(0) + beta(0)*j*delta(1) + beta(1)*k*delta(2));
		return acf;
	}

}

#endif // ACF_HH
