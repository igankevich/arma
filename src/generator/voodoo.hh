#ifndef VOODOO_HH
#define VOODOO_HH

#include <assert.h>      // for assert
#include <blitz/array.h> // for Range, toEnd, shape
#include <cstdlib>       // for abs
#include "types.hh"      // for Array2D, Shape3D

/**
\file
\author Ivan Gankevich
\date 2016-07-26

\brief
Non-optimised parts of the implementation.

\details
There exist more efficient ways to compute autoregressive model
coefficients that take into account autocovariance matrix structure, but they
seem complex and not wide-spread. So, we settled on "keep it simple" approach.
*/

namespace arma {

	namespace generator {

		namespace bits {

			template <class T>
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

			template <class T>
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
		}

		/**
		\brief Autocovariate matrix generator which uses least squares
		approximations to reduce size of autocovariate function grid.
		*/
		template <class T>
		struct AC_matrix_generator_LS {

			AC_matrix_generator_LS(const Array3D<T>& acf, const Shape3D& ar_order)
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
						bits::append_column_block(row, small_block(i, j));
					}
					bits::append_row_block(result, row);
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
						bits::append_column_block(row, small_block(i, j));
					}
					bits::append_row_block(result, row);
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

			const Array3D<T>& _acf;
			const Shape3D& _arorder;
		};

		/**
		\brief Autocovariate matrix generator that reduces the size of ACF to match
		AR model order.
		*/
		template <class T>
		struct AC_matrix_generator {

			AC_matrix_generator(const Array3D<T>& acf, const Shape3D& ar_order)
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
						bits::append_column_block(row, tmp);
					}
					bits::append_row_block(result, row);
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
						bits::append_column_block(row, tmp);
					}
					bits::append_row_block(result, row);
				}
				return result;
			}

		private:
			const Array3D<T>& _acf;
			const Shape3D& _arorder;
		};

		/**
		\brief Matrix generator for moving average model.
		*/
		template <class T>
		struct Tau_matrix_generator {

			Tau_matrix_generator(Array3D<T> tau) : _tau(tau) {}

			Array2D<T>
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

			Array2D<T>
			tau_matrix_block(int i0) {
				const int n = _tau.extent(1);
				Array2D<T> result;
				for (int i = 0; i < n; ++i) {
					Array2D<T> row;
					for (int j = 0; j < n; ++j) {
						Array2D<T> tmp(tau_matrix_block(i0, std::abs(i - j)) +
						               tau_matrix_block(i0, (i + j) % n));
						bits::append_column_block(row, tmp);
					}
					bits::append_row_block(result, row);
				}
				return result;
			}

			Array2D<T> operator()() {
				const int n = _tau.extent(0);
				Array2D<T> result;
				for (int i = 0; i < n; ++i) {
					Array2D<T> row;
					for (int j = 0; j < n; ++j) {
						Array2D<T> tmp(tau_matrix_block(std::abs(i - j)) +
						               tau_matrix_block((i + j) % n));
						bits::append_column_block(row, tmp);
					}
					bits::append_row_block(result, row);
				}
				return result;
			}

		private:
			Array3D<T> _tau;
		};

	}

}

#endif // VOODOO_HH
