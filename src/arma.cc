#include "arma.hh"

template <class T, int N>
void
arma::validate_process(blitz::Array<T, N> _phi) {
	if (blitz::product(_phi.shape()) <= 1) { return; }
	/// 1. Find roots of the polynomial
	/// \f$P_n(\Phi)=1-\Phi_1 x-\Phi_2 x^2 - ... -\Phi_n x^n\f$.
	blitz::Array<double, N> phi(_phi.shape());
	phi = -_phi;
	phi(0) = 1;
	/// 2. Trim leading zero terms.
	size_t real_size = 0;
	while (real_size < phi.numElements() && phi.data()[real_size] != 0.0) {
		++real_size;
	}
	blitz::Array<std::complex<double>, 1> result(real_size);
	gsl_poly_complex_workspace* w =
		gsl_poly_complex_workspace_alloc(real_size);
	int ret = gsl_poly_complex_solve(phi.data(), real_size, w,
									 (gsl_complex_packed_ptr)result.data());
	gsl_poly_complex_workspace_free(w);
	if (ret != GSL_SUCCESS) {
		std::cerr << "GSL error: " << gsl_strerror(ret) << '.' << std::endl;
		throw std::runtime_error(
			"Can not find roots of the polynomial to "
			"verify AR/MA model stationarity/invertibility.");
	}
	/// 3. Check if some roots do not lie outside unit circle.
	size_t num_bad_roots = 0;
	for (size_t i = 0; i < result.size(); ++i) {
		const double val = std::abs(result(i));
		/// Some AR coefficients are close to nought and polynomial
		/// solver can produce noughts due to limited numerical
		/// precision. So we filter val=0 as well.
		if (!(val > 1.0 || val == 0.0)) {
			++num_bad_roots;
			std::cerr << "Root #" << i << '=' << result(i) << std::endl;
		}
	}
	if (num_bad_roots > 0) {
		std::cerr << "No. of bad roots = " << num_bad_roots << std::endl;
		throw std::runtime_error(
			"AR/MA process is not stationary/invertible: some roots lie "
			"inside unit circle or on its borderline.");
	}
}

template void arma::validate_process<ARMA_REAL_TYPE,3>(
	blitz::Array<ARMA_REAL_TYPE,3> _phi
);
