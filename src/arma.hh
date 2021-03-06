#ifndef ARMA_HH
#define ARMA_HH

#include "types.hh"

/// @file
/// File with auxiliary subroutines.


/// Domain-specific classes and functions.
namespace arma {

	/// Check AR (MA) process stationarity (invertibility).
	template <class T, int N>
	void
	validate_process(blitz::Array<T, N> _phi);

	template <class T>
	T
	ACF_variance(const Array3D<T>& acf) {
		return acf(0,0,0);
	}

	/**
	   Compute white noise variance via the formula
	   \f[
	    \sigma_\alpha^2 = \frac{\gamma_0}{
	        \sum\limits_{i=0}^{n_1-1}
	        \sum\limits_{i=0}^{n_2-1}
	        \sum\limits_{k=0}^{n_3-1}
	        \theta_{i,j,k}^2
	    }
	   \f]
	   assuming \f$\theta_0 \equiv -1\f$.
	 */
	template <class T>
	T
	MA_white_noise_variance(const Array3D<T>& acf, const Array3D<T>& theta);

	/**
	   \brief Computes auto-covariance function of three-dimensional field.
	   \date 2018-01-30
	   \author Ivan Gankevich
	   \param[in] rhs symmetric three-dimensional field

	   Uses the following formula. Does not subtract mean value. Does not divide
	   by the variance.
	   \f[
	    \gamma_{i,j,k} =
	        \frac{1}{n_1 n_2 n_3}
	        \sum\limits_{i_1=0}^{n_1-1}
	        \sum\limits_{i_1=0}^{n_2-1}
	        \sum\limits_{k_1=0}^{n_3-1}
	        \zeta_{i_1,j_1,k_1}
	        \zeta_{(i_1+i) \bmod n_1,(j_1+i) \bmod n_2,(k_1+k) \bmod n_3}
	   \f]
	   Assumes, that the field is symmetric and periodic in each dimension.
	   Uses fast Fourier transforms to speed up the computation:
	   \f[
	     \gamma = \mathcal{F}^{-1}\left\{
		 	\left| \mathcal{F}\left\{\zeta\right\} \right|^2
		 \right\}.
	   \f]
	 */
	template <class T>
	Array3D<T>
	auto_covariance(const Array3D<T>& rhs);

}

#endif // ARMA_HH
