#include "yule_walker.hh"

#include <cassert>
#include <iostream>

#include "linalg.hh"

namespace {

	inline blitz::TinyVector<int,1>
	vector_shape(int a) {
		return blitz::shape(3*a*a + 3*a + 1);
	}

	template <class T>
	inline blitz::Array<T,1>
	rho_vector(int i, int j, int k, int d, blitz::Array<T,3> acf) {
		#define ACF(x,y,z) acf(std::abs(x), std::abs(y), std::abs(z))
		typedef blitz::Array<T,1> vector_type;
		using blitz::Range;
		vector_type rho_ijkd(vector_shape(d));
		/// Compute the first element.
		rho_ijkd(0) = ACF(i-d, j-d, k-d);
		std::clog << "rho_{" << i << "," << j << "," << k << "," << d <<"}(0)="
			<< rho_ijkd(0) << std::endl;
		int offset = 1;
		/// Compute row vectors \f$\rho_{i,j,k;d,e}\f$.
		for (int e=1; e<=d; ++e) {
			vector_type rho_ijkde(6*e);
			int off = 0;
			for (int idx=0; idx<e; ++idx) {
				rho_ijkde(off+idx) = ACF(i-d+e, j-d+idx, k-d);
			}
			off += e;
			for (int idx=0; idx<e; ++idx) {
				rho_ijkde(off+idx) = ACF(i-d+e-idx, j-d+e, k-d);
			}
			off += e;
			for (int idx=0; idx<e; ++idx) {
				rho_ijkde(off+idx) = ACF(i-d, j-d+e, k-d+idx);
			}
			off += e;
			for (int idx=0; idx<e; ++idx) {
				rho_ijkde(off+idx) = ACF(i-d, j-d+e-idx, k-d+e);
			}
			off += e;
			for (int idx=0; idx<e; ++idx) {
				rho_ijkde(off+idx) = ACF(i-d+idx, j-d, k-d+e);
			}
			off += e;
			for (int idx=0; idx<e; ++idx) {
				rho_ijkde(off+idx) = ACF(i-d+e, j-d, k-d+e-idx);
			}
			off += e;
			assert(off == rho_ijkde.extent(0));
			assert(off == rho_ijkd.extent(0)-1);
			rho_ijkd(Range(offset,offset+off-1)) = rho_ijkde;
			offset += off;
		}
		#undef ACF
		return rho_ijkd;
	}

	inline blitz::TinyVector<int,2>
	matrix_shape(int a, int b) {
		return blitz::shape(3*a*a + 3*a + 1, 3*b*b + 3*b + 1);
	}

	inline blitz::TinyVector<int,2>
	submatrix_shape(int a, int b) {
		return blitz::shape(6*a, 3*b*b + 3*b + 1);
	}

	template <class T>
	inline blitz::Array<T,2>
	R_matrix(int a, int b, blitz::Array<T,3> acf) {
		typedef blitz::Array<T,2> matrix_type;
		using blitz::Range;
		using blitz::shape;
		matrix_type R_ab(matrix_shape(a, b));
		/// Compute the first row.
		R_ab(0, Range::all()) = rho_vector(a, a, a, b, acf);
		/// Compute submatrices \f$\R_{a,b;e}\f$.
		int offset = 1;
		for (int e=1; e<=a; ++e) {
			matrix_type R_abe(submatrix_shape(a, b));
			int off = 0;
			/// Compute \f$\R_{a,b;e}^{(1)}\f$.
			for (int idx=0; idx<e; ++idx) {
				R_abe(off + idx, Range::all()) = rho_vector(a-e, a-idx, a, b, acf);
			}
			off += e;
			/// Compute \f$\R_{a,b;e}^{(2)}\f$.
			for (int idx=0; idx<e; ++idx) {
				R_abe(off + idx, Range::all()) = rho_vector(a-e+idx, a-e, a, b, acf);
			}
			off += e;
			/// Compute \f$\R_{a,b;e}^{(3)}\f$.
			for (int idx=0; idx<e; ++idx) {
				R_abe(off + idx, Range::all()) = rho_vector(a, a-e, a-idx, b, acf);
			}
			off += e;
			/// Compute \f$\R_{a,b;e}^{(4)}\f$.
			for (int idx=0; idx<e; ++idx) {
				R_abe(off + idx, Range::all()) = rho_vector(a, a-e+idx, a-e, b, acf);
			}
			off += e;
			/// Compute \f$\R_{a,b;e}^{(5)}\f$.
			for (int idx=0; idx<e; ++idx) {
				R_abe(off + idx, Range::all()) = rho_vector(a-idx, a, a-e, b, acf);
			}
			off += e;
			/// Compute \f$\R_{a,b;e}^{(6)}\f$.
			for (int idx=0; idx<e; ++idx) {
				R_abe(off + idx, Range::all()) = rho_vector(a-e, a, a-e+idx, b, acf);
			}
			off += e;
			assert(off == R_abe.extent(0));
			R_ab(Range(offset, offset+off-1), Range::all()) = R_abe;
			offset += off;
		}
		return R_ab;
	}

}

template <class T>
void
arma::solve_yule_walker(Array3D<T> acf, const T variance0, const int max_order) {
	typedef Array2D<T> matrix_type;
	using blitz::Range;
	/// Initial stage.
	matrix_type R_1_1 = R_matrix(1, 1, acf);
	matrix_type R_1_0 = R_matrix(1, 0, acf);
	matrix_type R_0_1 = R_matrix(0, 1, acf);
	std::clog << "R_1_1=" << R_1_1 << std::endl;
	std::clog << "R_1_0=" << R_1_0 << std::endl;
	std::clog << "R_0_1=" << R_0_1 << std::endl;
	matrix_type R_sup_1_1(R_1_1.shape());
	R_sup_1_1 = R_1_1;
	linalg::inverse(R_sup_1_1);
	std::clog << "R_1_1^{-1}=" << R_sup_1_1 << std::endl;
	std::clog << "R_1_1*R_1_1^{-1}=" << linalg::operator*(R_1_1, R_sup_1_1) << std::endl;
	matrix_type P_2_1 = linalg::operator*(R_sup_1_1, R_1_0);
	std::clog << "P_2_1=" << P_2_1 << std::endl;
	T variance = variance0*(T(1) - linalg::dot(R_0_1, P_2_1));
	std::clog << "variance=" << variance << std::endl;
	for (int l=2; l<max_order; ++l) {

	}
}

template void
arma::solve_yule_walker(
	Array3D<ARMA_REAL_TYPE> acf,
	const ARMA_REAL_TYPE,
	const int max_order
);

