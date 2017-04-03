#ifndef LINALG_HH
#define LINALG_HH

#include <cassert>
#include <blitz/array.h>
#include <lapacke/lapacke.h>
#include <openblas/cblas.h>

/**
\file
\author Ivan Gankevich
\date 2016-07-26
\brief Various linear algebra subroutines.
*/

/// Various linear algebra subroutines.
namespace linalg {

	template <class T>
	using Matrix = blitz::Array<T, 2>;
	template <class T>
	using Vector = blitz::Array<T, 1>;

	namespace bits {
		template<
			class T,
			class Vec,
			void (*gemv_ptr)(
				OPENBLAS_CONST enum CBLAS_ORDER order,
				OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,
				OPENBLAS_CONST blasint m,
				OPENBLAS_CONST blasint n,
				OPENBLAS_CONST T alpha,
				OPENBLAS_CONST T *a,
				OPENBLAS_CONST blasint lda,
				OPENBLAS_CONST T *x,
				OPENBLAS_CONST blasint incx,
				OPENBLAS_CONST T beta,
				T *y,
				OPENBLAS_CONST blasint incy
			)
		>
		Vec
		gemv(linalg::Matrix<T> lhs, Vec rhs) {
			const int m = lhs.rows(), n = lhs.cols();
			Vec result(rhs.shape());
			result = 0;
			gemv_ptr(CblasRowMajor, CblasNoTrans, m, n, T(1), lhs.data(), m,
			     rhs.data(), 1, T(0), result.data(), 1);
			return result;
		}

	}

	/**
	\brief Solve linear system \f$A x=b\f$ via Cholesky decomposition.
	\param[in]    A input matrix (lhs).
	\param[in,out] b input and output vector (rhs).
	*/
	template <class T>
	void
	cholesky(Matrix<T>& A, Vector<T>& b);

	template <class T>
	void
	inverse(Matrix<T>& A);

	template <int N>
	blitz::Array<float,N>
	operator*(Matrix<float> lhs, blitz::Array<float,N> rhs) {
		return bits::gemv<float,blitz::Array<float,N>,cblas_sgemv>(lhs, rhs);
	}

	template <int N>
	blitz::Array<double,N>
	operator*(Matrix<double> lhs, blitz::Array<double,N> rhs) {
		return bits::gemv<double,blitz::Array<double,N>,cblas_dgemv>(lhs, rhs);
	}

	template <class T>
	bool
	is_symmetric(Matrix<T>& rhs);

	/**
	\brief Use Cholesky decomposition to determine
	if matrix \p lhs is positive definite.
	*/
	template <class T>
	bool
	is_positive_definite(Matrix<T>& rhs);

	template <class T>
	bool
	is_toeplitz(Matrix<T>& rhs);

	/**
	\brief Make matrix square via least squares.

	\param[in] P input matrix (lhs).
	\param[in] p input vector (rhs).
	\param[out] A output matrix (lhs).
	\param[out] b output vector (rhs).
	*/
	template <class T>
	void
	least_squares(
		const Matrix<T>& P,
		const Vector<T>& p,
		Matrix<T>& A,
		Vector<T>& b
	);

	/**
	\brief Least squares interpolation.

	\param  x,y interpolation nodes.
	\param  n   no. of interpolation coefficients.
	\return     interpolation coefficients.
	*/
	template <class T>
	Vector<T>
	interpolate(const Vector<T>& x, const Vector<T>& y, const int n);

	/**
	\brief Solve equation \f$f(x)=0\f$ via bisection method.
	\param a left margin (\f$x_0\f$).
	\param b right margin (\f$x_1\f$).
	\param func lhs of equation (\f$f(x)\f$).
	\return solution to \f$f(x)=0\f$.
	*/
	template <class T, class F>
	T
	bisection(T a, T b, F func, T eps, int max_iter = 30) {
		T c, fc;
		int i = 0;
		do {
			c = T(0.5) * (a + b);
			fc = func(c);
			if (func(a) * fc < T(0)) b = c;
			if (func(b) * fc < T(0)) a = c;
			i++;
		} while (i < max_iter && (b - a) > eps && std::abs(fc) > eps);
		return c;
	}
}

#endif // LINALG_HH
