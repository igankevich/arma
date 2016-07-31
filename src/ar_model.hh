#ifndef AR_MODEL_HH
#define AR_MODEL_HH

#include <algorithm>  // for min, copy_n
#include <cassert>    // for assert
#include <complex>    // for complex, abs, operator<<
#include <cstdlib>    // for size_t
#include <functional> // for function
#include <iostream>   // for operator<<, basic_ostream, clog, endl
#include <fstream>    // for operator<<, basic_ostream, clog, endl
#include <stdexcept>  // for runtime_error

#include <blitz/array.h>     // for Range, toEnd, RectDomain, Array
#include <gsl/gsl_complex.h> // for gsl_complex_packed_ptr
#include <gsl/gsl_errno.h>   // for gsl_strerror, ::GSL_SUCCESS
#include <gsl/gsl_poly.h>    // for gsl_poly_complex_solve, gsl_poly_com...

#include "linalg.hh" // for cholesky, is_positive_definite, is_s...
#include "types.hh"  // for size3, Array3D, Array2D, Array1D, Zeta
#include "voodoo.hh" // for AC_matrix_generator, AC_matrix_gener...

/// @file
/// File with subroutines for AR model, Yule-Walker equations
/// and some others.

namespace autoreg {

	template <class T>
	struct Autoregressive_model {

		Autoregressive_model(Array3D<T> acf, size3 order, bool do_least_squares)
		    : _acf(acf), _order(order),
		      _phi(compute_coefficients(_acf, do_least_squares)) {}

		T
		white_noise_variance() {
			blitz::RectDomain<3> subdomain(size3(0, 0, 0), _phi.shape() - 1);
			return _acf(0, 0, 0) - blitz::sum(_phi * _acf(subdomain));
		}

		/**
		Generate wavy surface realisation.
		*/
		void operator()(Zeta<T>& zeta) {
			const size3 fsize = _phi.shape();
			const size3 zsize = zeta.shape();
			const int t1 = zsize[0];
			const int x1 = zsize[1];
			const int y1 = zsize[2];
			for (int t = 0; t < t1; t++) {
				for (int x = 0; x < x1; x++) {
					for (int y = 0; y < y1; y++) {
						const int m1 = std::min(t + 1, fsize[0]);
						const int m2 = std::min(x + 1, fsize[1]);
						const int m3 = std::min(y + 1, fsize[2]);
						T sum = 0;
						for (int k = 0; k < m1; k++)
							for (int i = 0; i < m2; i++)
								for (int j = 0; j < m3; j++)
									sum += _phi(k, i, j) *
									       zeta(t - k, x - i, y - j);
						zeta(t, x, y) += sum;
					}
				}
			}
		}

		/// Check AR process stationarity.
		void
		validate() {
			/// Find roots of the polynomial
			/// \f$P_n(\Phi)=1-\Phi_1 x-\Phi_2 x^2 - ... -\Phi_n x^n\f$.
			Array3D<double> phi(_phi.shape());
			phi = -_phi;
			phi(0) = 1;
			Array1D<std::complex<double>> result(phi.size());
			gsl_poly_complex_workspace* w =
			    gsl_poly_complex_workspace_alloc(phi.size());
			int ret =
			    gsl_poly_complex_solve(phi.data(), phi.size(), w,
			                           (gsl_complex_packed_ptr)result.data());
			gsl_poly_complex_workspace_free(w);
			if (ret != GSL_SUCCESS) {
				std::clog << "GSL error: " << gsl_strerror(ret) << '.'
				          << std::endl;
				throw std::runtime_error(
				    "Can not find roots of the polynomial to "
				    "verify AR model stationarity.");
			}
			/// Check if some roots do not lie outside unit circle.
			size_t num_bad_roots = 0;
			for (size_t i = 0; i < result.size(); ++i) {
				const double val = std::abs(result(i));
				/// Some AR coefficients are close to nought and polynomial
				/// solver can produce noughts due to limited numerical
				/// precision. So we filter val=0 as well.
				if (!(val > 1.0 || val == 0)) {
					++num_bad_roots;
					std::clog << "Root #" << i << '=' << result(i) << std::endl;
				}
			}
			if (num_bad_roots > 0) {
				std::clog << "No. of bad roots = " << num_bad_roots
				          << std::endl;
				throw std::runtime_error(
				    "AR process is not stationary: some roots lie "
				    "inside unit circle or on its borderline.");
			}
		}

	private:
		Array3D<T>
		compute_coefficients(Array3D<T> acf, bool do_least_squares) {

			if (_order(0) > acf.extent(0) || _order(1) > acf.extent(1) ||
			    _order(2) > acf.extent(2)) {
				std::clog << "AR model order is larger than ACF "
				             "size:\n\tAR model "
				             "order = "
				          << _order << "\n\tACF size = " << acf.shape()
				          << std::endl;
				throw std::runtime_error("bad AR model order");
			}

			acf = acf / acf(0, 0, 0);

			using blitz::Range;
			using blitz::toEnd;
			// normalise Array3D to prevent big numbers when multiplying
			// matrices
			std::function<Array2D<T>()> generator;
			if (do_least_squares) {
				generator = AC_matrix_generator_LS<T>(acf, _order);
			} else {
				generator = AC_matrix_generator<T>(acf, _order);
			}
			Array2D<T> acm = generator();
			{
				std::ofstream out("acm");
				out << acm;
			}
			const int m = acm.rows() - 1;

			/**
			eliminate the first equation and move the first column of the
			remaining
			matrix to the right-hand side of the system
			*/
			Array1D<T> rhs(m);
			rhs = acm(Range(1, toEnd), 0);
			//{ std::ofstream out("rhs"); out << rhs; }

			// lhs is the autocovariance matrix without first
			// column and row
			Array2D<T> lhs(blitz::shape(m, m));
			lhs = acm(Range(1, toEnd), Range(1, toEnd));
			//{ std::ofstream out("lhs"); out << lhs; }

			assert(lhs.extent(0) == m);
			assert(lhs.extent(1) == m);
			assert(rhs.extent(0) == m);
			assert(linalg::is_symmetric(lhs));
			assert(linalg::is_positive_definite(lhs));
			linalg::cholesky(lhs, rhs);
			Array3D<T> phi(_order);
			assert(phi.numElements() == rhs.numElements() + 1);
			phi(0, 0, 0) = 0;
			std::copy_n(rhs.data(), rhs.numElements(), phi.data() + 1);
			{
				std::ofstream out("ar_coefs");
				out << phi;
			}
			return phi;
		}

		Array3D<T> _acf;
		size3 _order;
		Array3D<T> _phi;
	};
}

#endif // AR_MODEL_HH
