#ifndef AUTOREG_HH
#define AUTOREG_HH

#include <algorithm>             // for min, any_of, copy_n, for_each, generate
#include <cassert>               // for assert
#include <chrono>                // for duration, steady_clock, steady_clock...
#include <cmath>                 // for isnan
#include <cstdlib>               // for abs
#include <functional>            // for bind
#include <iostream>              // for operator<<, cerr, endl
#include <fstream>               // for ofstream
#include <random>                // for mt19937, normal_distribution
#include <stdexcept>             // for runtime_error

#include <blitz/array.h>         // for Array, Range, shape, any

#include "sysv.hh"               // for sysv
#include "types.hh"              // for size3, ACF, AR_coefs, Zeta, Array2D
#include "voodoo.hh"             // for generate_AC_matrix

/// @file
/// File with subroutines for AR model, Yule-Walker equations
/// and some others.

namespace autoreg {

	template<class T>
	ACF<T>
	standing_wave_ACF(const Vec3<T>& delta, const size3& acf_size) {

		// guessed
//		T alpha = 0.06;
//		T beta = 0.8;
//		T gamm = 1.0;

		/// from Mathematica
		T alpha = 0.394279;
		T beta = 0.885028;
		T gamm = 0.0106085 * 100;

		ACF<T> acf(acf_size);
		blitz::firstIndex t;
		blitz::secondIndex x;
		blitz::thirdIndex y;
		acf = gamm
			* blitz::exp(-alpha*(t*delta[0] + x*delta[1] + y*delta[2]))
	 		* blitz::cos(beta * t * delta[0])
	 		* blitz::cos(beta * x * delta[1])
	 		* blitz::cos(beta * y * delta[2])
			;
		return acf;
	}

	template<class T>
	ACF<T>
	propagating_wave_ACF(const Vec3<T>& delta, const size3& acf_size) {
		/// values from Mathematica
//		T alpha = 0.177652;
//		T beta = -0.105817;
//		T gamm = 0.109203;
		/// values from Mathematica
//		T alpha = 1.32802;
//		T beta = 0.340238;
//		T gamm = 0.878945;

		// exp(-0.2*(|x| + |y| + |t|)
//		T alpha = 1.87405;
//		T beta = 3.04553e-9;
//		T gamm = 1.98831;

		/// from Mathematica
		T alpha = 0.163215;
		T beta = -0.0659017;
		T gamm = 0.0222666 * 100;

		ACF<T> acf(acf_size);
		blitz::firstIndex t;
		blitz::secondIndex x;
		blitz::thirdIndex y;
		/*
		acf = gamm
			* blitz::exp(-alpha*(
				t*delta[0]
				+ x*delta[1]
				+ y*delta[2]
			))
	 		* blitz::cos(beta*(
				 t*delta[0]
				 + x*delta[1]
				 + y*delta[2]
			));
		*/
		acf = gamm
			* blitz::exp(-alpha * (t*delta[0] + x*delta[1] + y*delta[2]))
			* blitz::cos(beta * (t*delta[0] + x*delta[1] + y*delta[2]))
			;
		return acf;
	}

	template<class T>
	T white_noise_variance(const AR_coefs<T>& ar_coefs, const ACF<T>& acf) {
		return acf(0,0,0) - blitz::sum(ar_coefs * acf);
	}

	template<class T>
	T ACF_variance(const ACF<T>& acf) {
		return acf(0,0,0);
	}

	/// Удаление участков разгона из реализации.
	template<class T>
	Zeta<T>
	trim_zeta(const Zeta<T>& zeta2, const size3& zsize) {
		using blitz::Range;
		using blitz::toEnd;
		size3 zsize2 = zeta2.shape();
		return zeta2(
			Range(zsize2(0) - zsize(0), toEnd),
			Range(zsize2(1) - zsize(1), toEnd),
			Range(zsize2(2) - zsize(2), toEnd)
		);
	}

	template<class T>
	bool is_stationary(AR_coefs<T>& phi) {
		return !blitz::any(blitz::abs(phi) > T(1));
	}

	template<class T>
	AR_coefs<T>
	clamp_coefficients(AR_coefs<T>& phi) {
		return AR_coefs<T>(blitz::where(blitz::abs(phi) > T(1), blitz::abs(phi)/phi, phi));
	}

	void
	dummy_matrix() {
		size3 sz(3, 3, 3);
		ACF<size3> acf(sz);
		for (int i=0; i<sz[0]; ++i) {
			for (int j=0; j<sz[1]; ++j) {
				for (int k=0; k<sz[2]; ++k) {
					acf(i,j,k) = size3(i,j,k);
				}
			}
		}
		const int m = acf.numElements()-1;
		Array2D<size3> acm = generate_AC_matrix(acf);
		{ std::ofstream out("acm_test"); out << acm; }
		using blitz::Range;
		using blitz::toEnd;
		Array1D<size3> rhs(m);
		rhs = acm(Range(1, toEnd), 0);
		{ std::ofstream out("rhs_test"); out << rhs; }

		// lhs is the autocovariance matrix without first
		// column and row
		Array2D<size3> lhs(blitz::shape(m,m));
		lhs = acm(Range(1, toEnd), Range(1, toEnd));
		{ std::ofstream out("lhs_test"); out << lhs; }

		AR_coefs<size3> phi(acf.shape());
		assert(phi.numElements() == rhs.numElements() + 1);
		phi(0,0,0) = 0;
		std::copy_n(rhs.data(), rhs.numElements(), phi.data()+1);
		{ std::ofstream out("ar_coefs_test"); out << phi; }
	}

	template<class T>
	AR_coefs<T>
	compute_AR_coefs(const ACF<T>& acf) {
		//dummy_matrix();
		using blitz::Range;
		using blitz::toEnd;
		const int m = acf.numElements()-1;
		Array2D<T> acm = generate_AC_matrix(acf);
		{ std::ofstream out("acm"); out << acm; }

		/**
		eliminate the first equation and move the first column of the remaining
		matrix to the right-hand side of the system
		*/
		Array1D<T> rhs(m);
		rhs = acm(Range(1, toEnd), 0);
		//{ std::ofstream out("rhs"); out << rhs; }

		// lhs is the autocovariance matrix without first
		// column and row
		Array2D<T> lhs(blitz::shape(m,m));
		lhs = acm(Range(1, toEnd), Range(1, toEnd));
		//{ std::ofstream out("lhs"); out << lhs; }

		assert(lhs.extent(0) == m);
		assert(lhs.extent(1) == m);
		assert(rhs.extent(0) == m);
		sgesv<T>(m, 1, lhs.data(), m, rhs.data(), m);
		AR_coefs<T> phi(acf.shape());
		assert(phi.numElements() == rhs.numElements() + 1);
		phi(0,0,0) = 0;
		std::copy_n(rhs.data(), rhs.numElements(), phi.data()+1);
		{ std::ofstream out("ar_coefs"); out << phi; }
//		phi = clamp_coefficients(phi);
		if (!is_stationary(phi)) {
			std::cerr << "phi.shape() = " << phi.shape() << std::endl;
			std::for_each(
				phi.begin(),
				phi.end(),
				[] (T val) {
					if (std::abs(val) > T(1)) {
						std::cerr << val << std::endl;
					}
				}
			);
//			throw std::runtime_error("AR process is not stationary, i.e. |phi| > 1");
		}
		return phi;
	}

	template<class T>
	bool
	isnan(T rhs) noexcept {
		return std::isnan(rhs);
	}

	/// Генерация белого шума по алгоритму Вихря Мерсенна и
	/// преобразование его к нормальному распределению по алгоритму Бокса-Мюллера.
	template<class T>
	Zeta<T>
	generate_white_noise(const size3& size, const T variance) {
		if (variance < T(0)) {
			throw std::runtime_error("variance is less than zero");
		}

		// инициализация генератора
		std::mt19937 generator;
		#if !defined(DISABLE_RANDOM_SEED)
		generator.seed(std::chrono::steady_clock::now().time_since_epoch().count());
		#endif
		std::normal_distribution<T> normal(T(0), std::sqrt(variance));

		// генерация и проверка
		Zeta<T> eps(size);
		std::generate(std::begin(eps), std::end(eps), std::bind(normal, generator));
		if (std::any_of(std::begin(eps), std::end(eps), &::autoreg::isnan<T>)) {
			throw std::runtime_error("white noise generator produced some NaNs");
		}
		return eps;
	}

	/// Генерация отдельных частей реализации волновой поверхности.
	template<class T>
	void generate_zeta(const AR_coefs<T>& phi, Zeta<T>& zeta) {
		Zeta<T> eps(zeta.shape());
		eps = zeta;
		zeta = 0;
		const size3 fsize = phi.shape();
		const size3 zsize = zeta.shape();
		const int t1 = zsize[0];
		const int x1 = zsize[1];
		const int y1 = zsize[2];
		for (int t=0; t<t1; t++) {
			for (int x=0; x<x1; x++) {
				for (int y=0; y<y1; y++) {
					const int m1 = std::min(t+1, fsize[0]);
					const int m2 = std::min(x+1, fsize[1]);
					const int m3 = std::min(y+1, fsize[2]);
					T sum = 0;
					for (int k=0; k<m1; k++)
						for (int i=0; i<m2; i++)
							for (int j=0; j<m3; j++)
								sum += phi(k, i, j)*zeta(t-k, x-i, y-j);
					zeta(t, x, y) = sum + eps(t, x, y);
				}
			}
		}
	}

	template<class T, int N>
	T mean(const blitz::Array<T,N>& rhs) {
		return blitz::sum(rhs) / rhs.numElements();
	}

	template<class T, int N>
	T variance(const blitz::Array<T,N>& rhs) {
		assert(rhs.numElements() > 0);
		const T m = mean(rhs);
		return blitz::sum(blitz::pow(rhs-m, 2)) / (rhs.numElements() - 1);
	}

}

#endif // AUTOREG_HH
