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
	standing_wave_ACF(const Vec3D<T>& delta, const Shape3D& acf_size) {

		// guessed
		T alpha = 0.06;
		T beta = 0.8;
		T gamm = 1.0;
		// TODO gamma=1 is needed for nonlinear transform

		// from mathematica
//		T alpha = 2.31906, beta = -5.49873, gamm = 0.0680413;

		Array3D<T> acf(acf_size);
		blitz::firstIndex t;
		blitz::secondIndex x;
		blitz::thirdIndex y;
		acf = gamm * blitz::exp(-alpha * (2 * t * delta[0] + x * delta[1] +
		                                  y * delta[2])) *
		      blitz::cos(2 * beta * t * delta[0]) *
		      blitz::cos(beta * x * delta[1]) *
		      blitz::cos(0 * beta * y * delta[2]);
		return acf;
	}

	template <class T>
	Array3D<T>
	propagating_wave_ACF(const Vec3D<T>& delta, const Shape3D& acf_size) {

		// guessed
//		T alpha = 1.5;
//		T gamm = 1.0;

		// from mathematica
		T alpha = 0.42, beta = -1.8, gamm = 1.0;

		Array3D<T> acf(acf_size);
		blitz::firstIndex i;
		blitz::secondIndex j;
		blitz::thirdIndex k;

		acf = gamm * blitz::exp(-alpha *
		                        (i * delta[0] + j * delta[1] + k * delta[2])) *
		      blitz::cos(-beta * i * delta[0] + beta * j * delta[1] +
		                 0*beta * k * delta[2]);
		return acf;
	}

}

#endif // ACF_HH
