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
#include "types.hh"  // for size3, Array3D, Array2D, Array1D, Array3D
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
		acf_variance() const {
			return _acf(0, 0, 0);
		}

		Array3D<T>
		coefficients() const {
			return _phi;
		}

		const size3&
		order() const {
			return _order;
		}

		T
		white_noise_variance() const {
			return white_noise_variance(_phi);
		}

		T
		white_noise_variance(Array3D<T> phi) const {
			blitz::RectDomain<3> subdomain(size3(0, 0, 0), phi.shape() - 1);
			return _acf(0, 0, 0) - blitz::sum(phi * _acf(subdomain));
		}

		void
		validate() const {
			validate_process(_phi);
		}

		/**
		Generate wavy surface realisation.
		*/
		Array3D<T> operator()(Array3D<T> zeta) {
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
			return zeta;
		}

		void
		determine_coefficients(bool do_least_squares) {
			// determine_coefficients_iteratively();
			determine_coefficients_old(do_least_squares);
		}

		void
		determine_coefficients_old(bool do_least_squares) {

			using blitz::all;

			if (!all(_order <= _acf.shape())) {
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
			if (_phi.numElements() > 1) {
				_phi(0, 0, 0) = 0;
			}
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
			using blitz::all;
			using blitz::isfinite;
			using blitz::sum;
			using blitz::RectDomain;
			const size3 _0(0, 0, 0);
			Array3D<T> r(_acf / _acf(0, 0, 0));
			Array3D<T> phi0(_order), phi1(_order);
			phi0 = 0;
			phi1 = 0;
			const int max_order = _order(0);
			//			phi0(0, 0, 0) = r(0, 0, 0);
			for (int p = 1; p < max_order; ++p) {
				const size3 order(p + 1, p + 1, p + 1);
				/// In three dimensions there are many "last" coefficients. We
				/// collect all their indices into a container to iterate over
				/// them.
				std::vector<size3> indices;
				// for (int i = 0; i < p; ++i) indices.emplace_back(i, p, p);
				// for (int i = 0; i < p; ++i) indices.emplace_back(p, i, p);
				// for (int i = 0; i < p; ++i) indices.emplace_back(p, p, i);
				indices.emplace_back(p, p, p);
				/// Compute coefficients on all three borders.
				for (const size3& idx : indices) {
					const RectDomain<3> sub1(_0, idx), rsub1(idx, _0);
					const T sum1 = sum(phi0(sub1) * r(rsub1));
					const T sum2 = sum(phi0(sub1) * r(sub1));
					phi0(idx) = (r(idx) - sum1) / (T(1) - sum2);
				}
				phi1 = phi0;
				/// Compute all other coefficients.
				{
					using namespace blitz::tensor;
					const size3 idx(p, p, p);
					const RectDomain<3> sub(_0, idx), rsub(idx, _0);
					phi1(sub) = phi0(sub) - phi1(p, p, p) * phi0(rsub);
				}
				phi0 = phi1;

				/// Validate white noise variance.
				const T var_wn = white_noise_variance(phi1);
				if (!std::isfinite(var_wn)) {
					std::clog << __FILE__ << ':' << __LINE__ << ':' << __func__
					          << ": bad white noise variance = " << var_wn
					          << std::endl;
					std::clog << "Indices: \n";
					std::copy(indices.begin(), indices.end(),
					          std::ostream_iterator<size3>(std::clog, "\n"));
					std::clog << std::endl;
					RectDomain<3> subdomain(_0, order - 1);
					std::clog << "phi1 = \n" << phi1(subdomain) << std::endl;
					throw std::runtime_error("bad white noise variance");
				}
#ifndef NDEBUG
				/// Print solver state.
				std::clog << __func__ << ':' << "Iteration=" << p
				          << ", var_wn=" << var_wn << std::endl;
#endif

				if (!all(isfinite(phi1))) {
					std::clog << __FILE__ << ':' << __LINE__ << ':' << __func__
					          << ": bad coefficients = \n" << phi1 << std::endl;
					throw std::runtime_error("bad AR model coefficients");
				}
			}
		}

		Array3D<T> _acf;
		size3 _order;
		Array3D<T> _phi;
	};
}

#endif // AR_MODEL_HH
