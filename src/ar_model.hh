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

#include <blitz/array.h> // for Range, toEnd, RectDomain, Array

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

		void
		validate() {
			validate_process(_phi);
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
