#ifndef AUTOREG_HH
#define AUTOREG_HH

#include <algorithm>  // for min, any_of, copy_n, for_each, generate
#include <cassert>    // for assert
#include <chrono>     // for duration, steady_clock, steady_clock...
#include <cmath>      // for isnan
#include <cstdlib>    // for abs
#include <functional> // for bind
#include <iostream>   // for operator<<, cerr, endl
#include <fstream>    // for ofstream
#include <random>     // for mt19937, normal_distribution
#include <stdexcept>  // for runtime_error
#include <complex>

#include <blitz/array.h> // for Array, Range, shape, any

#include "types.hh"  // for size3, ACF, AR_coefs, Zeta, Array2D
#include "voodoo.hh" // for generate_AC_matrix
#include "linalg.hh"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_poly.h>

/// @file
/// File with subroutines for AR model, Yule-Walker equations
/// and some others.

/// Domain-specific classes and functions.
namespace autoreg {

	template <class T>
	ACF<T>
	standing_wave_ACF(const Vec3<T>& delta, const size3& acf_size) {

		// guessed
		T alpha = 0.06;
		T beta = 0.8;
		T gamm = 5.0;

		/// from Mathematica
		//		T alpha = 0.394279;
		//		T beta = 0.885028;
		//		T gamm = 0.0106085 * 100;

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
		/// values from Mathematica
		T alpha = 0.177652;
		T beta = -0.105817;
		T gamm = 0.109203 * 10;
		/// values from Mathematica
		//		T alpha = 1.32802;
		//		T beta = 0.340238;
		//		T gamm = 0.878945;

		// exp(-0.2*(|x| + |y| + |t|)
		//		T alpha = 1.87405;
		//		T beta = 3.04553e-9;
		//		T gamm = 1.98831;

		/// from Mathematica
		//		T alpha = 0.163215;
		//		T beta = 0.0659017 / 16;
		//		T gamm = 2;

		ACF<T> acf(acf_size);
		blitz::firstIndex t;
		blitz::secondIndex x;
		blitz::thirdIndex y;
		acf =
		    gamm * blitz::exp(-alpha * (t * delta[0] + 4 * x * delta[1] +
		                                y * delta[2])) *
		    blitz::cos(beta * (t * delta[0] + 0 * x * delta[1] + y * delta[2]));
		return acf;
	}

	template <class T>
	ACF<T>
	propagating_wave_ACF_2(const Vec3<T>& delta, const size3& acf_size) {

		// first plain wave
		T a1 = 1;
		T k1 = 0.5;
		T w1 = 0.1;

		// second plain wave
		T a2 = 1.1;
		T k2 = 0.4;
		T w2 = 0.2;

		T damp_x = 0.01;
		T damp_y = 0.01;
		T damp_t = 0.30;

		ACF<T> acf(acf_size);
		//		blitz::firstIndex t;
		//		blitz::secondIndex x;
		//		blitz::thirdIndex y;

		const Vec3<T> max = (acf_size - 1) * delta;
		for (int i = 0; i < acf_size[0]; ++i) {
			for (int j = 0; j < acf_size[1]; ++j) {
				for (int k = 0; k < acf_size[2]; ++k) {
					const T t = i * delta[0];
					const T x = j * delta[1];
					const T y = k * delta[2];
					const T r = std::sqrt(x * x + y * y);
					acf(i, j, k) = std::exp(-damp_t * i * delta[0] -
					                        damp_x * j * delta[1] -
					                        damp_y * k * delta[2]) *
					               (a1 * std::cos(T(2) * M_PI *
					                              (k1 * r + k1 * r - w1 * t)) +
					                a2 * std::cos(T(2) * M_PI *
					                              (k2 * r + k2 * r - w2 * t)));
					if (r >= max(1) || i == acf_size[0] - 1) {
						acf(i, j, k) = 0;
					}
				}
			}
		}

		/*
		acf = blitz::exp(
		    -damp_t*t*delta[0]
		    -damp_x*x*delta[1]
		    -damp_y*y*delta[2]
		) * (
		    a1*blitz::cos(T(2)*M_PI * (k1*x*delta[1] + k1*y*delta[2] -
		w1*t*delta[0])) +
		    a2*blitz::cos(T(2)*M_PI * (k2*x*delta[1] + k2*y*delta[2] -
		w2*t*delta[0]))
		);
		*/

		return acf;
	}

	template <class T>
	ACF<T>
	propagating_wave_ACF_3(const Vec3<T>& delta, const size3& acf_size) {

		/// from Mathematica
		T alpha = 2.3;
		T beta = 0;
		T gamm = 5.5;

		ACF<T> acf(acf_size);
		blitz::firstIndex i;
		blitz::secondIndex j;
		blitz::thirdIndex k;

		acf = gamm * blitz::exp(-alpha * (i * delta[0] + 0.1 * j * delta[1] +
		                                  k * delta[2])) *
		      blitz::cos(beta * (i * delta[0] + j * delta[1] + k * delta[2]));
		return acf;
	}

	template <class T>
	T
	white_noise_variance(const AR_coefs<T>& ar_coefs, const ACF<T>& acf) {
		return acf(0, 0, 0) - blitz::sum(ar_coefs * acf);
	}

	template <class T>
	T
	ACF_variance(const ACF<T>& acf) {
		return acf(0, 0, 0);
	}

	/// Удаление участков разгона из реализации.
	template <class T>
	Zeta<T>
	trim_zeta(const Zeta<T>& zeta2, const size3& zsize) {
		using blitz::Range;
		using blitz::toEnd;
		size3 zsize2 = zeta2.shape();
		return zeta2(Range(zsize2(0) - zsize(0), toEnd),
		             Range(zsize2(1) - zsize(1), toEnd),
		             Range(zsize2(2) - zsize(2), toEnd));
	}

	template <class T>
	void
	check_stationarity(AR_coefs<T>& phi_in) {
		/// Find roots of the polynomial
		/// \f$P_n(\Phi)=1-\Phi_1 x-\Phi_2 x^2 - ... -\Phi_n x^n\f$.
		AR_coefs<double> phi(phi_in.shape());
		phi = -phi_in;
		phi(0) = 1;
		Array1D<std::complex<double>> result(phi.size());
		gsl_poly_complex_workspace* w =
		    gsl_poly_complex_workspace_alloc(phi.size());
		int ret = gsl_poly_complex_solve(phi.data(), phi.size(), w,
		                                 (gsl_complex_packed_ptr)result.data());
		gsl_poly_complex_workspace_free(w);
		if (ret != GSL_SUCCESS) {
			throw std::runtime_error("Can not find roots of the polynomial to "
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
			std::clog << "No. of bad roots = " << num_bad_roots << std::endl;
			throw std::runtime_error(
			    "AR process is not stationary: some roots lie "
			    "inside unit circle or on its borderline.");
		}
	}

	int
	adhoc_t(int i, size3 s) {
		return ((i / s[2]) / s[1]) % s[0];
	}
	int
	adhoc_x(int i, size3 s) {
		return (i / s[2]) % s[1];
	}
	int
	adhoc_y(int i, size3 s) {
		return i % s[2];
	}

	template <class T>
	AR_coefs<T>
	compute_AR_coefs(const ACF<T>& acf, const size3& ar_order) {
		if (ar_order(0) > acf.extent(0) || ar_order(1) > acf.extent(1) ||
		    ar_order(2) > acf.extent(2)) {
			std::stringstream msg;
			msg << "AR order is larger than ACF size:\n\tAR order = "
			    << ar_order << "\n\tACF size = " << acf.shape();
			throw std::runtime_error(msg.str());
		}

		using blitz::Range;
		using blitz::toEnd;
		AC_matrix_generator<T> generator(acf, ar_order);
		Array2D<T> acm = generator();
		{
			std::ofstream out("acm");
			out << acm;
		}
//		const int m = acf.numElements() - 1;
		const int m = acm.rows() - 1;

		/**
		eliminate the first equation and move the first column of the remaining
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

		// alternative Yule-Walker matrix constructor
		//		Array2D<T> lhs(blitz::shape(m, m));
		//		Array1D<T> rhs(m);
		//		size3 s = acf.shape();
		//		for (int i=0; i<m; i++) {
		//			for (int j=0; j<m; j++) {
		//				lhs(i,j) = acf(
		//					std::abs(adhoc_t(i+1, s) - adhoc_t(j+1, s)),
		//				    std::abs(adhoc_x(i+1, s) - adhoc_x(j+1, s)),
		//				    std::abs(adhoc_y(i+1, s) - adhoc_y(j+1, s))
		//				);
		//			}
		//			rhs(i) = acf(
		//				adhoc_t(i+1, s),
		//				adhoc_x(i+1, s),
		//				adhoc_y(i+1, s)
		//			);
		//		}

		assert(lhs.extent(0) == m);
		assert(lhs.extent(1) == m);
		assert(rhs.extent(0) == m);
		assert(linalg::is_symmetric(lhs));
		assert(linalg::is_positive_definite(lhs));
		linalg::cholesky(lhs, rhs);
		//		sgesv<T>(m, 1, lhs.data(), m, rhs.data(), m);
		AR_coefs<T> phi(ar_order);
		assert(phi.numElements() == rhs.numElements() + 1);
		phi(0, 0, 0) = 0;
		std::copy_n(rhs.data(), rhs.numElements(), phi.data() + 1);
		{
			std::ofstream out("ar_coefs");
			out << phi;
		}
//		check_stationarity(phi);
		return phi;
	}

	template <class T>
	bool
	isnan(T rhs) noexcept {
		return std::isnan(rhs);
	}

	template <class T>
	bool
	isinf(T rhs) noexcept {
		return std::isinf(rhs);
	}

	/// Генерация белого шума по алгоритму Вихря Мерсенна и
	/// преобразование его к нормальному распределению по алгоритму
	/// Бокса-Мюллера.
	template <class T>
	Zeta<T>
	generate_white_noise(const size3& size, T variance) {
		if (variance < T(0)) {
			throw std::runtime_error("variance is less than zero");
		}

		// инициализация генератора
		std::mt19937 generator;
#if !defined(DISABLE_RANDOM_SEED)
		generator.seed(
		    std::chrono::steady_clock::now().time_since_epoch().count());
#endif
		std::normal_distribution<T> normal(T(0), std::sqrt(variance));

		// генерация и проверка
		Zeta<T> eps(size);
		std::generate(std::begin(eps), std::end(eps),
		              std::bind(normal, generator));
		if (std::any_of(std::begin(eps), std::end(eps), &::autoreg::isnan<T>)) {
			throw std::runtime_error(
			    "white noise generator produced some NaNs");
		}
		if (std::any_of(std::begin(eps), std::end(eps), &::autoreg::isinf<T>)) {
			throw std::runtime_error(
			    "white noise generator produced infinite numbers");
		}
		return eps;
	}

	/// Генерация отдельных частей реализации волновой поверхности.
	template <class T>
	void
	generate_zeta(AR_coefs<T>& phi, Zeta<T>& zeta) {
		const size3 fsize = phi.shape();
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
								sum += phi(k, i, j) * zeta(t - k, x - i, y - j);
					zeta(t, x, y) += sum;
				}
			}
		}
	}

	template <class T, int N>
	T
	mean(const blitz::Array<T, N>& rhs) {
		assert(rhs.numElements() > 0);
		return blitz::sum(rhs) / rhs.numElements();
	}

	template <class T, int N>
	T
	variance(const blitz::Array<T, N>& rhs) {
		assert(rhs.numElements() > 1);
		const T m = mean(rhs);
		return blitz::sum(blitz::pow(rhs - m, 2)) / (rhs.numElements() - 1);
	}
}

#endif // AUTOREG_HH
