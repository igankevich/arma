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

		Autoregressive_model(Array3D<T> acf, size3 order)
		    : _acf(acf), _order(order), _phi(_order) {}

		T
		white_noise_variance() {
			return white_noise_variance(_phi);
		}

		T
		white_noise_variance(Array3D<T> phi) {
			blitz::RectDomain<3> subdomain(size3(0, 0, 0), phi.shape() - 1);
			return _acf(0, 0, 0) - blitz::sum(phi * _acf(subdomain));
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

		void
		determine_coefficients(bool do_least_squares) {
			determine_coefficients_iteratively();
		}

		void
		determine_coefficients_old(bool do_least_squares) {

			if (_order(0) > _acf.extent(0) || _order(1) > _acf.extent(1) ||
			    _order(2) > _acf.extent(2)) {
				std::clog << "AR model order is larger than ACF "
				             "size:\n\tAR model "
				             "order = "
				          << _order << "\n\tACF size = " << _acf.shape()
				          << std::endl;
				throw std::runtime_error("bad AR model order");
			}

			//_acf = _acf / _acf(0, 0, 0);

			using blitz::Range;
			using blitz::toEnd;
			// normalise Array3D to prevent big numbers when multiplying
			// matrices
			std::function<Array2D<T>()> generator;
			if (do_least_squares) {
				generator = AC_matrix_generator_LS<T>(_acf, _order);
			} else {
				generator = AC_matrix_generator<T>(_acf, _order);
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
			assert(_phi.numElements() == rhs.numElements() + 1);
			_phi(0, 0, 0) = 0;
			std::copy_n(rhs.data(), rhs.numElements(), _phi.data() + 1);
			{
				std::ofstream out("ar_coefs");
				out << _phi;
			}
		}

	private:
		/**
		Darbin algorithm. Partial autocovariation function \f$\phi_{k,j}\f$,
		where k --- AR process order, j --- coefficient index.
		*/
		void
		determine_coefficients_iteratively() {
			Array3D<T> r(_acf / _acf(0, 0, 0));
			Array3D<T> phi0(_order), phi1(_order);
			phi0 = 0;
			phi1 = 0;
			const int max_order = _order(0);
			phi0(0, 0, 0) = r(0, 0, 0);
			for (int p = 1; p < max_order; ++p) {
				size3 order(p + 1, p + 1, p + 1);
				T sum1 = 0;
				for (int i = 0; i < p; ++i) {
					for (int j = 0; j < p; ++j) {
						for (int k = 0; k < p; ++k) {
							sum1 += phi0(i, j, k) *
							        r(p - i - 1, p - j - 1, p - k - 1);
						}
					}
				}
				blitz::RectDomain<3> sub2(size3(0, 0, 0),
				                          size3(p - 1, p - 1, p - 1));
				T sum2 = blitz::sum(phi0(sub2) * r(sub2));
				/// Compute the last coefficient.
				T last_coef = (r(p, p, p) - sum1) / (T(1) - sum2);
				phi1(p, p, p) = last_coef;
				for (int i = 0; i < p + 1; ++i) {
					for (int j = 0; j < p + 1; ++j) {
						for (int k = 0; k < p + 1; ++k) {
							phi1(i, j, k) =
							    phi0(i, j, k) -
							    last_coef * phi0(p - i, p - j, p - k);
						}
					}
				}
				/// Restore the last coefficient.
				phi1(p, p, p) = last_coef;
				phi0 = phi1;

				std::clog << __func__
				          << ": var_wn=" << white_noise_variance(phi1)
				          << std::endl;

				if (!blitz::all(blitz::isfinite(phi1))) {
					std::clog << "sum1 = " << sum1 << std::endl;
					std::clog << "sum2 = " << sum2 << std::endl;
					std::clog << "last = " << last_coef << std::endl;
					blitz::RectDomain<3> subdomain(size3(0, 0, 0), order - 1);
					std::clog << "phi1 = \n" << phi1(subdomain) << std::endl;
					break;
				}
			}
		}

		Array3D<T> _acf;
		size3 _order;
		Array3D<T> _phi;
	};
}

#endif // AR_MODEL_HH
