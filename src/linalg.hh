#ifndef LINALG_HH
#define LINALG_HH

#include <cassert>
#include <blitz/array.h>
#include <lapacke/lapacke.h>

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

	/**
	\brief Solve linear system \f$A x=b\f$ via Cholesky decomposition.
	\param[in]    A input matrix (lhs).
	\param[in,out] b input and output vector (rhs).
	*/
	template <class T>
	void
	cholesky(Matrix<T>& A, Vector<T>& b) {
		assert(A.extent(0) == A.extent(1));
		assert(A.extent(0) == b.extent(0));
		Vector<lapack_int> ipiv(A.rows());
		LAPACKE_ssysv(LAPACK_ROW_MAJOR, 'U', A.rows(), 1, A.data(), A.cols(),
		              ipiv.data(), b.data(), 1);
	}

	template <class T>
	bool
	is_symmetric(Matrix<T>& rhs) {
		using blitz::firstDim;
		using blitz::secondDim;
		return blitz::all(rhs == rhs.transpose(secondDim, firstDim));
	}

	/**
	\brief Use Cholesky decomposition to determine
	if matrix \p lhs is positive definite.
	*/
	template <class T>
	bool
	is_positive_definite(Matrix<T>& rhs) {
		assert(rhs.rows() == rhs.cols());
		using blitz::Range;
		Matrix<T> L(rhs.shape());
		L = 0;
		const int n = rhs.rows();
		for (int i = 0; i < n; ++i) {
			T sum = blitz::sum(blitz::pow2(L(i, Range(0, i - 1))));
			if (rhs(i, i) < sum) { return false; }
			L(i, i) = std::sqrt(rhs(i, i) - sum);
			for (int j = 0; j < i; ++j) {
				sum = blitz::sum(L(i, Range(0, j - 1)) * L(j, Range(0, j - 1)));
				L(i, j) = (rhs(i, j) - sum) / L(j, j);
			}
		}
		return true;
	}

	template <class T>
	bool
	is_toeplitz(Matrix<T>& rhs) {
		const int nrows = rhs.rows();
		const int ncols = rhs.cols();
		// check if rhs(i,j) == rhs(i-1,j-1)
		for (int i = 1; i < nrows; ++i) {
			for (int j = 1; j < ncols; ++j) {
				if (rhs(i, j) != rhs(i - 1, j - 1)) { return false; }
			}
		}
		return true;
	}

	/**
	\brief Make matrix square via least squares.

	\param[in] P input matrix (lhs).
	\param[in] p input vector (rhs).
	\param[out] A output matrix (lhs).
	\param[out] b output vector (rhs).
	*/
	template <class T>
	void
	least_squares(const Matrix<T>& P, const Vector<T>& p, Matrix<T>& A,
	              Vector<T>& b) {
		assert(P.extent(0) == p.extent(0));
		assert(A.extent(0) == A.extent(1));
		assert(A.extent(0) == b.extent(0));
		assert(A.extent(1) == P.extent(1));
		assert(A.extent(0) <= P.extent(0));
		const int N = p.numElements();
		const int n = b.numElements();
		for (int k = 0; k < n; ++k) {
			b(k) = 0;
			for (int j = 0; j < n; ++j) { A(k, j) = 0; }
			for (int i = 0; i < N; ++i) {
				for (int j = 0; j < n; ++j) { A(k, j) += P(i, k) * P(i, j); }
				b(k) += P(i, k) * p(i);
			}
		}
	}

	/**
	\brief Least squares interpolation.

	\param  x,y interpolation nodes.
	\param  n   no. of interpolation coefficients.
	\return     interpolation coefficients.
	*/
	template <class T>
	Vector<T>
	interpolate(const Vector<T>& x, const Vector<T>& y, const int n) {
		assert(x.numElements() == y.numElements());
		Vector<T> a(n);
		const int N = x.numElements();
		Matrix<T> A(blitz::shape(N, n));
		for (int i = 0; i < N; i++) {
			for (int k = 0; k < n; k++) { A(i, k) = std::pow(x(i), k); }
		}
		Vector<T> b2(n);
		Vector<T> A2(blitz::shape(n, n));
		least_squares(A, y, A2, b2, n, N);
		cholesky(A2, b2, n, a);
		return a;
	}

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
