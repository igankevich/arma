#ifndef AUTOREG_HH
#define AUTOREG_HH

#include <random>
#include <iterator>
#include <functional>
#include <sstream>
#include <cassert>

#include <blitz/array.h>

#include "stdx.hh"
#include "vector_n.hh"

/// @file
/// File with subroutines for AR model, Yule-Walker equations
/// and some others.

namespace autoreg {

	template<class T> using ACF = blitz::Array<T,3>;
	template<class T> using AR_coefs = blitz::Array<T,3>;
	template<class T> using Matrix = blitz::Array<T,2>;
	template<class T> using Zeta = blitz::Array<T,3>;

	template<class T>
	ACF<T>
	approx_acf(T alpha, T beta, T gamm, const Vec3<T>& delta, const size3& acf_size) {
		ACF<T> acf(acf_size);
		blitz::firstIndex t;
		blitz::secondIndex x;
		blitz::thirdIndex y;
		acf = gamm
			* blitz::exp(-alpha * (t*delta[0] + x*delta[1] + y*delta[2]))
	 		* blitz::cos(beta * (t*delta[0] + x*delta[1] + y*delta[2]));
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
	Matrix<T>
	AC_matrix_block(const ACF<T>& acf, int i0, int j0) {
		const int n = acf.extent(2);
		Matrix<T> block(blitz::shape(n, n));
		for (int i=0; i<n; ++i) {
			for (int j=0; j<n; ++j) {
				block(i, j) = acf(i0, j0, std::abs(i-j));
			}
		}
		return block;
	}

	template<class T>
	void
	append_column_block(Matrix<T>& lhs, const Matrix<T>& rhs) {
		if (lhs.numElements() == 0) {
			lhs.resize(rhs.shape());
			lhs = rhs;
		} else {
			using blitz::Range;
			assert(lhs.rows() == rhs.rows());
			const int old_cols = lhs.columns();
			lhs.resizeAndPreserve(lhs.rows(), old_cols + rhs.columns());
			lhs(Range::all(), Range(old_cols, blitz::toEnd)) = rhs;
		}
	}

	template<class T>
	void
	append_row_block(Matrix<T>& lhs, const Matrix<T>& rhs) {
		if (lhs.numElements() == 0) {
			lhs.resize(rhs.shape());
			lhs = rhs;
		} else {
			using blitz::Range;
			assert(lhs.columns() == rhs.columns());
			const int old_rows = lhs.rows();
			lhs.resizeAndPreserve(old_rows + rhs.rows(), lhs.columns());
			lhs(Range(old_rows, blitz::toEnd), Range::all()) = rhs;
		}
	}

	template<class T>
	Matrix<T>
	AC_matrix_block(const ACF<T>& acf, int i0) {
		const int n = acf.extent(1);
		Matrix<T> result;
		for (int i=0; i<n; ++i) {
			Matrix<T> row;
			for (int j=0; j<n; ++j) {
				append_column_block(row, AC_matrix_block(acf, i0, std::abs(i-j)));
			}
			append_row_block(result, row);
		}
		return result;
	}

	template<class T>
	Matrix<T>
	generate_AC_matrix(const ACF<T>& acf) {
		const int n = acf.extent(0);
		Matrix<T> result;
		for (int i=0; i<n; ++i) {
			Matrix<T> row;
			for (int j=0; j<n; ++j) {
				append_column_block(row, AC_matrix_block(acf, std::abs(i-j)));
			}
			append_row_block(result, row);
		}
		return result;
	}

	template<class T>
	AR_coefs<T>
	compute_AR_coefs(const ACF<T>& acf) {
		AR_coefs<T> phi(acf.shape());
		phi = acf;
		const size_t m = phi.numElements();
		Matrix<T> lhs = generate_AC_matrix(acf);
		sysv<T>('U', m, 1, lhs.data(), m, phi.data(), m);
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
			throw std::runtime_error("AR process is not stationary (|phi| > 1)");
		}
		return phi;
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
		std::normal_distribution<T> normal(T(0), variance);

		// генерация и проверка
		Zeta<T> eps(size);
		std::generate(std::begin(eps), std::end(eps), std::bind(normal, generator));
		if (std::any_of(std::begin(eps), std::end(eps), &stdx::isnan<T>)) {
			throw std::runtime_error("white noise generator produced some NaNs");
		}
		return eps;
	}

	/// Генерация отдельных частей реализации волновой поверхности.
	template<class T>
	void generate_zeta(const AR_coefs<T>& phi,
					   const size3& fsize,
					   const size3& zsize,
					   Zeta<T>& zeta)
	{
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
					zeta(t, x, y) += sum;
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
		return blitz::sum(blitz::pow(rhs, 2)) / rhs.numElements();
	}

}

#endif // AUTOREG_HH
