#include "linalg.hh"

namespace {

	namespace bits {

		template<
			class T,
			lapack_int (*sysv)(int matrix_layout, char uplo, lapack_int n,
                               lapack_int nrhs, T* a, lapack_int lda,
                               lapack_int* ipiv, T* b, lapack_int ldb)
		>
		void
		cholesky(linalg::Matrix<T>& A, linalg::Vector<T>& b) {
			assert(A.extent(0) == A.extent(1));
			assert(A.extent(0) == b.extent(0));
			linalg::Vector<lapack_int> ipiv(A.rows());
			sysv(LAPACK_ROW_MAJOR, 'U', A.rows(), 1, A.data(), A.cols(),
			     ipiv.data(), b.data(), 1);
		}

		template <
			class T,
			lapack_int (*getrf)(
				int matrix_layout,
				lapack_int m,
				lapack_int n,
				T* a,
				lapack_int lda,
				lapack_int* ipiv
			),
			lapack_int (*getri)(
				int matrix_layout,
				lapack_int n,
				T* a,
				lapack_int lda,
				const lapack_int* ipiv
			)
		>
		void
		inverse(linalg::Matrix<T>& A) {
			const int m = A.rows(), n = A.cols();
			linalg::Vector<lapack_int> ipiv(std::min(m, n));
			getrf(LAPACK_ROW_MAJOR, m, n, A.data(), m, ipiv.data());
			getri(LAPACK_ROW_MAJOR, m, A.data(), m, ipiv.data());
		}

	}
}

namespace linalg {

	template <>
	void
	cholesky<float>(Matrix<float>& A, Vector<float>& b) {
		::bits::cholesky<float,LAPACKE_ssysv>(A, b);
	}

	template <>
	void
	cholesky<double>(Matrix<double>& A, Vector<double>& b) {
		::bits::cholesky<double,LAPACKE_dsysv>(A, b);
	}

	template <>
	void
	inverse(Matrix<float>& A) {
		::bits::inverse<float,LAPACKE_sgetrf,LAPACKE_sgetri>(A);
	}

	template <>
	void
	inverse(Matrix<double>& A) {
		::bits::inverse<double,LAPACKE_dgetrf,LAPACKE_dgetri>(A);
	}

}


template <class T>
bool
linalg::is_symmetric(Matrix<T>& rhs) {
	using blitz::firstDim;
	using blitz::secondDim;
	return blitz::all(rhs == rhs.transpose(secondDim, firstDim));
}

template <class T>
bool
linalg::is_positive_definite(Matrix<T>& rhs) {
	assert(rhs.rows() == rhs.cols());
	using blitz::Range;
	using blitz::sum;
	using blitz::pow2;
	using std::sqrt;
	Matrix<T> L(rhs.shape());
	L = 0;
	const int n = rhs.rows();
	for (int i = 0; i < n; ++i) {
		T s = sum(pow2(L(i, Range(0, i - 1))));
		if (rhs(i, i) < s) { return false; }
		L(i, i) = sqrt(rhs(i, i) - s);
		for (int j = 0; j < i; ++j) {
			s = sum(L(i, Range(0, j - 1)) * L(j, Range(0, j - 1)));
			L(i, j) = (rhs(i, j) - s) / L(j, j);
		}
	}
	return true;
}

template <class T>
bool
linalg::is_toeplitz(Matrix<T>& rhs) {
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

template <class T>
void
linalg::least_squares(
	const Matrix<T>& P,
	const Vector<T>& p,
	Matrix<T>& A,
	Vector<T>& b
) {
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

template <class T>
linalg::Vector<T>
linalg::interpolate(const Vector<T>& x, const Vector<T>& y, const int n) {
	assert(x.numElements() == y.numElements());
	const int N = x.numElements();
	Matrix<T> A(blitz::shape(N, n));
	for (int i = 0; i < N; i++) {
		for (int k = 0; k < n; k++) { A(i, k) = std::pow(x(i), k); }
	}
	Vector<T> b2(n);
	Matrix<T> A2(blitz::shape(n, n));
	least_squares(A, y, A2, b2);
	cholesky(A2, b2);
	return b2;
}


template bool linalg::is_symmetric<ARMA_REAL_TYPE>(Matrix<ARMA_REAL_TYPE>& rhs);
template bool linalg::is_positive_definite<ARMA_REAL_TYPE>(Matrix<ARMA_REAL_TYPE>& rhs);
template bool linalg::is_toeplitz<ARMA_REAL_TYPE>(Matrix<ARMA_REAL_TYPE>& rhs);

template linalg::Vector<ARMA_REAL_TYPE>
linalg::interpolate<ARMA_REAL_TYPE>(
	const Vector<ARMA_REAL_TYPE>& x,
	const Vector<ARMA_REAL_TYPE>& y,
	const int n
);

template void
linalg::least_squares<ARMA_REAL_TYPE>(
	const Matrix<ARMA_REAL_TYPE>& P,
	const Vector<ARMA_REAL_TYPE>& p,
	Matrix<ARMA_REAL_TYPE>& A,
	Vector<ARMA_REAL_TYPE>& b
);
