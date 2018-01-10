#include "yule_walker.hh"

#include <cassert>
#include <fstream>
#include <iostream>
#include <unordered_map>

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
			matrix_type R_abe(submatrix_shape(e, b));
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

	template <class T>
	inline blitz::Array<T,2>
	operator*(blitz::Array<T,2> lhs, blitz::Array<T,2> rhs) {
		return linalg::multiply(lhs, rhs);
	}

}

namespace std {

	template <>
	struct hash<std::tuple<int,int>>: public std::hash<int> {
		inline size_t
		operator()(const std::tuple<int,int>& rhs) const noexcept {
			typedef std::hash<int> base;
			return base::operator()(std::get<0>(rhs)) ^
				   base::operator()(std::get<1>(rhs));
		}
	};

	template <>
	struct hash<std::tuple<int,int,int>>: public std::hash<int> {
		inline size_t
		operator()(const std::tuple<int,int,int>& rhs) const noexcept {
			typedef std::hash<int> base;
			return base::operator()(std::get<0>(rhs)) ^
				   base::operator()(std::get<1>(rhs)) ^
				   base::operator()(std::get<2>(rhs));
		}
	};

}

template <class T>
void
arma::solve_yule_walker(Array3D<T> acf, const T variance0, const int max_order) {
	#define PHI(i,j,k) Phi[std::make_tuple(i,j,k)]
	typedef Array2D<T> matrix_type;
	typedef std::tuple<int,int,int> key_type;
	typedef std::unordered_map<key_type,matrix_type> map3d_type;
	using blitz::Range;
	using blitz::shape;
	/// Initial stage.
	map3d_type Phi;
	blitz::Array<matrix_type,2> Pi(shape(max_order+2,max_order+2));
	blitz::Array<matrix_type,2> beta(shape(max_order+2,max_order+2));
	blitz::Array<matrix_type,1> Theta(shape(max_order+2));
	matrix_type R_1_1 = R_matrix(1, 1, acf);
	matrix_type R_1_0 = R_matrix(1, 0, acf);
	matrix_type R_0_1 = R_matrix(0, 1, acf);
	std::clog << "R_1_1=" << R_1_1 << std::endl;
	std::clog << "R_1_0=" << R_1_0 << std::endl;
	std::clog << "R_0_1=" << R_0_1 << std::endl;
	matrix_type R_sup_1_1(R_1_1.shape());
	R_sup_1_1 = R_1_1;
	assert(linalg::is_symmetric(R_sup_1_1));
	linalg::inverse(R_sup_1_1);
	std::clog << "R_1_1^{-1}=" << R_sup_1_1 << std::endl;
//	std::clog << "R_1_1*R_1_1^{-1}=" << (R_1_1*R_sup_1_1) << std::endl;
	matrix_type Pi_2_1(R_sup_1_1*R_1_0);
	std::clog << "Pi_2_1=" << Pi_2_1 << std::endl;
	T lambda = T(1) - linalg::dot(R_0_1, Pi_2_1);
	T variance = variance0*lambda;
	std::clog << "variance=" << variance << std::endl;
	matrix_type Phi_2_1_0 = R_sup_1_1*R_matrix(1, 2, acf);
	Pi(2,1).reference(Pi_2_1);
	PHI(2,1,0).reference(Phi_2_1_0);
	for (int l=2; l<=max_order; ++l) {
		matrix_type R_l_l = R_matrix(l, l, acf);
		matrix_type R_l_0 = R_matrix(l, 0, acf);
		matrix_type sum1(R_l_l.shape());
		sum1 = 0;
		matrix_type sum2(R_l_0.shape());
		sum2 = 0;
		for (int m=1; m<=l-1; ++m) {
			matrix_type R_l_m(R_matrix(l, m, acf));
			sum1 += R_l_m*PHI(l,m,0);
			sum2 += R_l_m*Pi(l,m);
		}
		matrix_type Theta_l(R_l_l - sum1);
		matrix_type h_l(R_l_0 - sum2);
		std::clog << "h_l=" << h_l << std::endl;
		std::clog << "Theta_l=" << Theta_l << std::endl;
		linalg::inverse(Theta_l);
		Theta(l).reference(Theta_l);
//		{ std::ifstream("/tmp/z") >> Theta_l; }
		std::clog << "Theta_l^{-1}=" << Theta_l << std::endl;
		Pi(l+1,l).reference(Theta_l * h_l);
		std::clog << "Pi(l+1,l)=" << Pi(l+1,l) << std::endl;
		for (int a=1; a<=l-1; ++a) {
			Pi(l+1,a).reference(matrix_type(Pi(l,a) - PHI(l,a,0)*Pi(l+1,l)));
		}
		lambda -= linalg::dot(h_l, Pi(l+1,l));
		variance = variance0*lambda;
		for (int a=1; a<=l; ++a) {
			beta(l,a).reference(Pi(l+1,a));
			std::clog << "beta(l,a)=" << beta(l,a) << std::endl;
		}
		std::clog << "variance=" << variance << std::endl;
		if (l < max_order) {
			PHI(2,1,l-1).reference(matrix_type(R_sup_1_1*R_matrix(1, l+1, acf)));
			for (int m=2; m<=l; ++m) {
				matrix_type R_m_lp1(R_matrix(m,l+1,acf));
				matrix_type sum3(R_m_lp1.shape());
				sum3 = 0;
				for (int n=1; n<=m-1; ++n) {
					sum3 += R_matrix(m,n,acf)*PHI(m,n,l-m+1);
				}
				PHI(m+1,m,l-m).reference(matrix_type(Theta(m)*matrix_type(R_m_lp1 - sum3)));
				for (int a=1; a<=m-1; ++a) {
					std::clog << "a=" << a << std::endl;
					std::clog << "m=" << m << std::endl;
					std::clog << "l=" << l << std::endl;
					PHI(m+1,a,l-m).reference(matrix_type(
						PHI(m,a,l-m+1) - PHI(m,a,0)*PHI(m+1,m,l-m)
					));
				}
			}
		}
	}
	#undef PHI
}

template void
arma::solve_yule_walker(
	Array3D<ARMA_REAL_TYPE> acf,
	const ARMA_REAL_TYPE,
	const int max_order
);

