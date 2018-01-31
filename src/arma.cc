#include "arma.hh"

#ifndef NDEBUG
#include <fstream>
#endif

#include <gsl/gsl_complex.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_poly.h>

#include "fourier.hh"

template <class T, int N>
void
arma
::validate_process(blitz::Array<T, N> _phi) {
	typedef blitz::Array<std::complex<double>, 1> result_type;
	if (blitz::product(_phi.shape()) <= 1) {
		return;
	}
	/// 1. Find roots of the polynomial
	/// \f$P_n(\Phi)=1-\Phi_1 x-\Phi_2 x^2 - ... -\Phi_n x^n\f$.
	blitz::Array<double, N> phi(_phi.shape());
	phi = -_phi;
	phi(0) = 1;
	result_type result(phi.numElements());
	gsl_poly_complex_workspace* w =
		gsl_poly_complex_workspace_alloc(result.size());
	int ret =
		gsl_poly_complex_solve(
			phi.data(),
			result.size(),
			w,
			(gsl_complex_packed_ptr)result.data()
		);
	gsl_poly_complex_workspace_free(w);
	if (ret != GSL_SUCCESS) {
		std::cerr << "GSL error: " << gsl_strerror(ret) << '.' << std::endl;
		throw std::runtime_error(
				  "Can not find roots of the polynomial to "
				  "verify AR/MA model stationarity/invertibility."
		);
	}
	/// 2. Check if some roots do not lie outside the unit circle.
	int num_bad_roots = 0;
	const int nroots = result.size() - 1;
	for (int i=0; i<nroots; ++i) {
		const double val = std::abs(result(i));
		if (!(val > 1.0)) {
			++num_bad_roots;
			std::cerr << "Root #" << i << '=' << result(i) << std::endl;
		}
	}
	#ifndef NDEBUG
	typedef blitz::Array<double, 1> result_real_type;
	{ std::ofstream("poly") << phi; }
	{ std::ofstream("poly_roots") << result; }
	{ std::ofstream("poly_roots_abs") << result_real_type(blitz::abs(result)); }
	#endif
	if (num_bad_roots > 0) {
		std::cerr << "No. of bad roots = " << num_bad_roots << std::endl;
		throw std::runtime_error(
				  "AR/MA process is not stationary/invertible: some roots lie "
				  "inside unit circle or on its borderline."
		);
	}
}

template <class T>
T
arma
::MA_white_noise_variance(const Array3D<T>& acf, const Array3D<T>& theta) {
	using blitz::sum;
	using blitz::pow2;
	return ACF_variance(acf) / sum(pow2(theta));
}

template <class T>
arma::Array3D<T>
arma
::auto_covariance(const Array3D<T>& rhs) {
	using arma::apmath::Fourier_transform;
	using arma::apmath::Fourier_workspace;
	using blitz::abs;
	using blitz::pow2;
	using blitz::real;
	typedef std::complex<T> C;
	Fourier_transform<C,3> fft(rhs.shape());
	Fourier_workspace<C,3> workspace(rhs.shape());
	const int n = rhs.numElements();
	blitz::Array<C,3> cwave(rhs.shape());
	cwave = rhs;
	cwave = pow2(abs(fft.forward(cwave, workspace)));
	return blitz::Array<T,3>(
		real(fft.backward(cwave, workspace)) / n / n
	);
}

template void
arma
::validate_process<ARMA_REAL_TYPE,3>(blitz::Array<ARMA_REAL_TYPE,3> _phi);

template ARMA_REAL_TYPE
arma
::MA_white_noise_variance(
	const Array3D<ARMA_REAL_TYPE>& acf,
	const Array3D<ARMA_REAL_TYPE>& theta
);

template arma::Array3D<ARMA_REAL_TYPE>
arma
::auto_covariance(const Array3D<ARMA_REAL_TYPE>& rhs);
