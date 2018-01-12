#include "voodoo.hh"

#include "config.hh"

namespace {

	template <class T>
	arma::Array2D<T>
	multiply_matrices(arma::Array2D<T> lhs, arma::Array2D<T> rhs) {
		const int nrows = lhs.rows();
		const int ncols = rhs.columns();
		const int n = lhs.columns();
		arma::Array2D<T> result(blitz::shape(nrows, ncols));
		result = 0;
		for (int i = 0; i < nrows; ++i) {
			for (int j = 0; j < ncols; ++j) {
				for (int k = 0; k < n; ++k) {
					result(i, j) += lhs(i, k) * rhs(k, j);
				}
			}
		}
		assert(blitz::product(result.shape()) > 0);
		return result;
	}

	template <class T>
	void
	append_column_block(arma::Array2D<T>& lhs, const arma::Array2D<T>& rhs) {
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

	template <class T>
	void
	append_row_block(arma::Array2D<T>& lhs, const arma::Array2D<T>& rhs) {
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

}

template <class T>
arma::Array2D<T>
arma::generator::AC_matrix_generator<T>::
AC_matrix_block(int i0, int j0) {
	const int n = _arorder(2);
	Array2D<T> block(blitz::shape(n, n));
	for (int k = 0; k < n; ++k) {
		for (int j = 0; j < n; ++j) {
			block(k, j) = _acf(i0, j0, std::abs(k - j));
		}
	}
	return block;
}

template <class T>
arma::Array2D<T>
arma::generator::AC_matrix_generator<T>::
AC_matrix_block(int i0) {
	const int n = _arorder(1);
	Array2D<T> result;
	for (int i = 0; i < n; ++i) {
		Array2D<T> row;
		for (int j = 0; j < n; ++j) {
			Array2D<T> tmp = AC_matrix_block(i0, std::abs(i - j));
			append_column_block(row, tmp);
		}
		append_row_block(result, row);
	}
	return result;
}

template <class T>
arma::Array2D<T>
arma::generator::AC_matrix_generator<T>::
operator()() {
	const int n = _arorder(0);
	Array2D<T> result;
	for (int i = 0; i < n; ++i) {
		Array2D<T> row;
		for (int j = 0; j < n; ++j) {
			Array2D<T> tmp = AC_matrix_block(std::abs(i - j));
			append_column_block(row, tmp);
		}
		append_row_block(result, row);
	}
	return result;
}

template <class T>
arma::Array2D<T>
arma::generator::Tau_matrix_generator<T>::
tau_matrix_block(int i0, int j0) {
	const int n = _tau.extent(2);
	Array2D<T> block(blitz::shape(n, n));
	block = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n - i; ++j) {
			block(i, j) += _tau(i0, j0, (i + j) % n);
		}
	}
	for (int i = 0; i < n; ++i) {
		for (int j = i; j < n; ++j) {
			block(i, j) += _tau(i0, j0, std::abs(j - i));
		}
	}
	return block;
}

template <class T>
arma::Array2D<T>
arma::generator::Tau_matrix_generator<T>::
tau_matrix_block(int i0) {
	const int n = _tau.extent(1);
	Array2D<T> result;
	for (int i = 0; i < n; ++i) {
		Array2D<T> row;
		for (int j = 0; j < n; ++j) {
			Array2D<T> tmp(tau_matrix_block(i0, std::abs(i - j)) +
						   tau_matrix_block(i0, (i + j) % n));
			append_column_block(row, tmp);
		}
		append_row_block(result, row);
	}
	return result;
}

template <class T>
arma::Array2D<T>
arma::generator::Tau_matrix_generator<T>::
operator()() {
	const int n = _tau.extent(0);
	Array2D<T> result;
	for (int i = 0; i < n; ++i) {
		Array2D<T> row;
		for (int j = 0; j < n; ++j) {
			Array2D<T> tmp(tau_matrix_block(std::abs(i - j)) +
						   tau_matrix_block((i + j) % n));
			append_column_block(row, tmp);
		}
		append_row_block(result, row);
	}
	return result;
}

template class arma::generator::AC_matrix_generator<ARMA_REAL_TYPE>;
template class arma::generator::Tau_matrix_generator<ARMA_REAL_TYPE>;
