#ifndef ACF_HH
#define ACF_HH

#include "types.hh" // for ACF, Vec3, size3

/// @file
/// Mini-database of ACF approximations.

namespace autoreg {

	template <class T>
	ACF<T>
	standing_wave_ACF(const Vec3<T>& delta, const size3& acf_size) {

		// guessed
		T alpha = 0.06;
		T beta = 0.8;
		T gamm = 5.0;

		ACF<T> acf(acf_size);
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
	ACF<T>
	propagating_wave_ACF(const Vec3<T>& delta, const size3& acf_size) {

		// guessed
		T alpha = 1.5;
		T gamm = 5.0;

		ACF<T> acf(acf_size);
		blitz::firstIndex i;
		blitz::secondIndex j;
		blitz::thirdIndex k;

		acf = gamm * blitz::exp(-alpha * (i * delta[0] + j * delta[1] +
		                                  k * delta[2]));
		acf -= blitz::min(acf);
		return acf;
	}
}

#endif // ACF_HH
