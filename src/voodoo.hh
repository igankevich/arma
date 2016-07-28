#ifndef VOODOO_HH
#define VOODOO_HH

#include <assert.h>      // for assert
#include <cstdlib>       // for abs
#include <blitz/array.h> // for Array, Range, shape, any
#include "types.hh"      // for Array2D, ACF

namespace autoreg {

	/// Least squares.
	template <class T>
	struct AC_matrix_generator_LS {

		AC_matrix_generator_LS(const ACF<T>& acf, const size3& ar_order)
		    : _acf(acf), _arorder(ar_order) {}

		Array2D<T>
		AC_matrix_block(int i0, int j0) {
			const int m = _acf.extent(2);
			const int n = _arorder(2);
			Array2D<T> block(blitz::shape(n, n));
			block = 0;
			for (int k = 0; k < n; ++k) {
				for (int i = 0; i < m; ++i) {
					for (int j = 0; j < n; ++j) {
						block(k, j) += _acf(i0, j0, std::abs(i - k)) *
						               _acf(i0, j0, std::abs(i - j));
					}
				}
			}
			assert(blitz::product(block.shape()) > 0);
			return block;
		}

		Array2D<T>
		AC_matrix_block(int i0) {
			const int m = _acf.extent(1);
			const int n = _arorder(1);
			// pre-compute all matrix blocks
			Array2D<Array2D<T>> block(blitz::shape(m, n));
			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < n; ++j) {
					Array2D<T> tmp = AC_matrix_block(i0, std::abs(i - j));
					block(i, j).resize(tmp.shape());
					block(i, j) = tmp;
				}
			}
			// reduce matrix size via least squares fitting
			Array2D<Array2D<T>> small_block(blitz::shape(n, n));
			for (int k = 0; k < n; ++k) {
				for (int i = 0; i < m; ++i) {
					for (int j = 0; j < n; ++j) {
						Array2D<T> tmp =
						    multiply_matrices(block(i, k), block(i, j));
						small_block(k, j).resize(tmp.shape());
						small_block(k, j) = 0;
						small_block(k, j) += tmp;
					}
				}
			}
			// flatten block matrix
			Array2D<T> result;
			for (int i = 0; i < n; ++i) {
				Array2D<T> row;
				for (int j = 0; j < n; ++j) {
					append_column_block(row, small_block(i, j));
				}
				append_row_block(result, row);
			}
			return result;
		}

		Array2D<T> operator()() {
			const int m = _acf.extent(0);
			const int n = _arorder(0);
			// pre-compute all matrix blocks
			Array2D<Array2D<T>> block(blitz::shape(m, n));
			for (int i = 0; i < m; ++i) {
				for (int j = 0; j < n; ++j) {
					Array2D<T> tmp = AC_matrix_block(std::abs(i - j));
					block(i, j).resize(tmp.shape());
					block(i, j) = tmp;
				}
			}
			// reduce matrix size via least squares fitting
			Array2D<Array2D<T>> small_block(blitz::shape(n, n));
			for (int k = 0; k < n; ++k) {
				for (int i = 0; i < m; ++i) {
					for (int j = 0; j < n; ++j) {
						Array2D<T> tmp =
						    multiply_matrices(block(i, k), block(i, j));
						small_block(k, j).resize(tmp.shape());
						small_block(k, j) = 0;
						small_block(k, j) += tmp;
					}
				}
			}
			// flatten block matrix
			Array2D<T> result;
			for (int i = 0; i < n; ++i) {
				Array2D<T> row;
				for (int j = 0; j < n; ++j) {
					append_column_block(row, small_block(i, j));
				}
				append_row_block(result, row);
			}
			return result;
		}

	private:
		Array2D<T>
		multiply_matrices(Array2D<T> lhs, Array2D<T> rhs) {
			const int nrows = lhs.rows();
			const int ncols = rhs.columns();
			const int n = lhs.columns();
			Array2D<T> result(blitz::shape(nrows, ncols));
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

		void
		append_column_block(Array2D<T>& lhs, const Array2D<T>& rhs) {
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

		void
		append_row_block(Array2D<T>& lhs, const Array2D<T>& rhs) {
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

		const ACF<T>& _acf;
		const size3& _arorder;
	};

	/// Slicing
	template <class T>
	struct AC_matrix_generator {

		AC_matrix_generator(const ACF<T>& acf, const size3& ar_order)
		    : _acf(acf), _arorder(ar_order) {}

		Array2D<T>
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

		Array2D<T>
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

		Array2D<T> operator()() {
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

	private:
		void
		append_column_block(Array2D<T>& lhs, const Array2D<T>& rhs) {
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

		void
		append_row_block(Array2D<T>& lhs, const Array2D<T>& rhs) {
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

		const ACF<T>& _acf;
		const size3& _arorder;
	};
}

#endif // VOODOO_HH
