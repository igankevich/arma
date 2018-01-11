#include "yule_walker.hh"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <unordered_map>

#ifndef NDEBUG
#include <iostream>
#endif

#include "linalg.hh"

namespace {

	inline blitz::TinyVector<int,1>
	vector_shape(int a) {
		return blitz::shape(3*a*a + 3*a + 1);
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
	inline blitz::Array<T,1>
	rho_vector(int i, int j, int k, int d, const blitz::Array<T,3>& acf) {
		#define ACF(x,y,z) acf(std::abs(x), std::abs(y), std::abs(z))
		typedef blitz::Array<T,1> vector_type;
		using blitz::Range;
		vector_type rho_ijkd(vector_shape(d));
		/// Compute the first element.
		T* data = rho_ijkd.data();
		*data++ = ACF(i-d, j-d, k-d);
		/// Compute row vectors \f$\rho_{i,j,k;d,e}\f$.
		for (int e=1; e<=d; ++e) {
			for (int idx=0; idx<e; ++idx) {
				*data++ = ACF(i-d+e, j-d+idx, k-d);
			}
			for (int idx=0; idx<e; ++idx) {
				*data++ = ACF(i-d+e-idx, j-d+e, k-d);
			}
			for (int idx=0; idx<e; ++idx) {
				*data++ = ACF(i-d, j-d+e, k-d+idx);
			}
			for (int idx=0; idx<e; ++idx) {
				*data++ = ACF(i-d, j-d+e-idx, k-d+e);
			}
			for (int idx=0; idx<e; ++idx) {
				*data++ = ACF(i-d+idx, j-d, k-d+e);
			}
			for (int idx=0; idx<e; ++idx) {
				*data++ = ACF(i-d+e, j-d, k-d+e-idx);
			}
		}
		#undef ACF
		return rho_ijkd;
	}

	// For d=0.
	template <class T>
	inline T
	rho_scalar(int i, int j, int k, const blitz::Array<T,3>& acf) {
		#define ACF(x,y,z) acf(std::abs(x), std::abs(y), std::abs(z))
		return ACF(i, j, k);
		#undef ACF
	}

	template <class T>
	inline blitz::Array<T,2>
	R_matrix(int a, int b, const blitz::Array<T,3>& acf) {
		using blitz::Range;
		typedef blitz::Array<T,2> matrix_type;
		matrix_type R_ab(matrix_shape(a, b));
		/// Compute the first row.
		R_ab(0, Range::all()) = rho_vector(a, a, a, b, acf);
		/// Compute submatrices \f$\R_{a,b;e}\f$.
		int offset = 1;
		for (int e=1; e<=a; ++e) {
			/// Compute \f$\R_{a,b;e}^{(1)}\f$.
			for (int idx=0; idx<e; ++idx) {
				R_ab(offset++, Range::all()) = rho_vector(a-e, a-idx, a, b, acf);
			}
			/// Compute \f$\R_{a,b;e}^{(2)}\f$.
			for (int idx=0; idx<e; ++idx) {
				R_ab(offset++, Range::all()) = rho_vector(a-e+idx, a-e, a, b, acf);
			}
			/// Compute \f$\R_{a,b;e}^{(3)}\f$.
			for (int idx=0; idx<e; ++idx) {
				R_ab(offset++, Range::all()) = rho_vector(a, a-e, a-idx, b, acf);
			}
			/// Compute \f$\R_{a,b;e}^{(4)}\f$.
			for (int idx=0; idx<e; ++idx) {
				R_ab(offset++, Range::all()) = rho_vector(a, a-e+idx, a-e, b, acf);
			}
			/// Compute \f$\R_{a,b;e}^{(5)}\f$.
			for (int idx=0; idx<e; ++idx) {
				R_ab(offset++, Range::all()) = rho_vector(a-idx, a, a-e, b, acf);
			}
			/// Compute \f$\R_{a,b;e}^{(6)}\f$.
			for (int idx=0; idx<e; ++idx) {
				R_ab(offset++, Range::all()) = rho_vector(a-e, a, a-e+idx, b, acf);
			}
		}
		return R_ab;
	}

	// For b=0.
	template <class T>
	inline blitz::Array<T,1>
	R_vector_b0(int a, const blitz::Array<T,3>& acf) {
		using blitz::Range;
		using blitz::shape;
		typedef blitz::Array<T,1> vector_type;
		vector_type R_ab(vector_shape(a));
		T* data = R_ab.data();
		/// Compute the first row.
		*data++ = rho_scalar(a, a, a, acf);
		/// Compute submatrices \f$\R_{a,0;e}\f$.
		for (int e=1; e<=a; ++e) {
			/// Compute \f$\R_{a,0;e}^{(1)}\f$.
			for (int idx=0; idx<e; ++idx) {
				*data++ = rho_scalar(a-e, a-idx, a, acf);
			}
			/// Compute \f$\R_{a,0;e}^{(2)}\f$.
			for (int idx=0; idx<e; ++idx) {
				*data++ = rho_scalar(a-e+idx, a-e, a, acf);
			}
			/// Compute \f$\R_{a,0;e}^{(3)}\f$.
			for (int idx=0; idx<e; ++idx) {
				*data++ = rho_scalar(a, a-e, a-idx, acf);
			}
			/// Compute \f$\R_{a,0;e}^{(4)}\f$.
			for (int idx=0; idx<e; ++idx) {
				*data++ = rho_scalar(a, a-e+idx, a-e, acf);
			}
			/// Compute \f$\R_{a,0;e}^{(5)}\f$.
			for (int idx=0; idx<e; ++idx) {
				*data++ = rho_scalar(a-idx, a, a-e, acf);
			}
			/// Compute \f$\R_{a,0;e}^{(6)}\f$.
			for (int idx=0; idx<e; ++idx) {
				*data++ = rho_scalar(a-e, a, a-e+idx, acf);
			}
		}
		return R_ab;
	}

	// For a=0.
	template <class T>
	inline blitz::Array<T,1>
	R_vector_a0(int b, blitz::Array<T,3> acf) {
		return rho_vector(0, 0, 0, b, acf);
	}

	template <class T>
	inline blitz::Array<T,3>
	result_array(blitz::Array<blitz::Array<T,1>,1> beta, int order) {
		blitz::Array<T,3> result(blitz::shape(order, order, order));
		result = 0;
		result(0,0,0) = 0;
		for (int d=1; d<order; ++d) {
			const T* data = beta(d).data();
			result(d,d,d) = *data++;
			for (int e=1; e<=d; ++e) {
				for (int idx=0; idx<e; ++idx) {
					result(d-e,d-idx,d) = *data++;
				}
				for (int idx=0; idx<e; ++idx) {
					result(d-e+idx,d-e,d) = *data++;
				}
				for (int idx=0; idx<e; ++idx) {
					result(d,d-e,d-idx) = *data++;
				}
				for (int idx=0; idx<e; ++idx) {
					result(d,d-e+idx,d-e) = *data++;
				}
				for (int idx=0; idx<e; ++idx) {
					result(d-idx,d,d-e) = *data++;
				}
				for (int idx=0; idx<e; ++idx) {
					result(d-e,d,d-e+idx) = *data++;
				}
			}
		}
		return result;
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
arma::Array3D<T>
arma::solve_yule_walker(Array3D<T> acf, const T variance0, const int max_order) {
	#define PHI(i,j,k) Phi[std::make_tuple(i,j,k)]
	typedef blitz::Array<T,2> matrix_type;
	typedef blitz::Array<T,1> vector_type;
	typedef std::tuple<int,int,int> key_type;
	typedef std::unordered_map<key_type,matrix_type> map3d_type;
	using blitz::Range;
	using blitz::shape;
	/// Initial stage.
	map3d_type Phi;
	blitz::Array<vector_type,1> Pi_l(shape(max_order+2));
	blitz::Array<vector_type,1> Pi_lp1(shape(max_order+2));
	blitz::Array<matrix_type,1> Theta(shape(max_order+2));
	matrix_type R_1_1 = R_matrix(1, 1, acf);
	vector_type R_1_0 = R_vector_b0(1, acf);
	vector_type R_0_1 = R_vector_a0(1, acf);
//	std::clog << "R_1_1=" << R_1_1 << std::endl;
//	std::clog << "R_1_0=" << R_1_0 << std::endl;
//	std::clog << "R_0_1=" << R_0_1 << std::endl;
	matrix_type R_sup_1_1(R_1_1.shape());
	R_sup_1_1 = R_1_1;
	assert(linalg::is_symmetric(R_sup_1_1));
	linalg::inverse(R_sup_1_1);
//	std::clog << "R_1_1^{-1}=" << R_sup_1_1 << std::endl;
//	std::clog << "R_1_1*R_1_1^{-1}=" << (R_1_1*R_sup_1_1) << std::endl;
	vector_type Pi_2_1(linalg::multiply_mv(R_sup_1_1, R_1_0));
//	std::clog << "Pi_2_1=" << Pi_2_1 << std::endl;
	T lambda = T(1) - linalg::dot(R_0_1, Pi_2_1);
	T variance = variance0*lambda;
	#ifndef NDEBUG
	/// Print solver state.
	std::clog << __func__ << ':' << "order=" << 1
			  << ",variance=" << variance << std::endl;
	#endif
	Pi_l(1).reference(Pi_2_1);
	PHI(2,1,0).reference(matrix_type(R_sup_1_1*R_matrix(1, 2, acf)));
	for (int l=2; l<=max_order; ++l) {
		matrix_type R_l_l = R_matrix(l, l, acf);
		vector_type R_l_0 = R_vector_b0(l, acf);
		matrix_type sum1(R_l_l.shape());
		sum1 = 0;
		vector_type sum2(R_l_0.shape());
		sum2 = 0;
		for (int m=1; m<=l-1; ++m) {
			matrix_type R_l_m(R_matrix(l, m, acf));
			sum1 += R_l_m*PHI(l,m,0);
			sum2 += linalg::multiply_by_column_vector(R_l_m, Pi_l(m));
		}
		matrix_type Theta_l(R_l_l - sum1);
		vector_type h_l(R_l_0 - sum2);
//		std::clog << "h_l=" << h_l << std::endl;
//		std::clog << "Theta_l=" << Theta_l << std::endl;
		linalg::inverse(Theta_l);
		Theta(l).reference(Theta_l);
//		{ std::ifstream("/tmp/z") >> Theta_l; }
//		std::clog << "Theta_l^{-1}=" << Theta_l << std::endl;
		Pi_lp1(l).reference(linalg::multiply_mv(Theta_l, h_l));
//		std::clog << "Pi(l+1,l)=" << Pi_lp1(l) << std::endl;
		for (int a=1; a<=l-1; ++a) {
			Pi_lp1(a).reference(vector_type(
				Pi_l(a) - linalg::multiply_by_column_vector(PHI(l,a,0), Pi_lp1(l))
			));
		}
		lambda -= linalg::dot(h_l, Pi_lp1(l));
		variance = variance0*lambda;
		#ifndef NDEBUG
		/// Print solver state.
		std::clog << __func__ << ':' << "order=" << l
				  << ",variance=" << variance << std::endl;
		#endif
		if (l < max_order) {
			PHI(2,1,l-1).reference(R_sup_1_1*R_matrix(1, l+1, acf));
			for (int m=2; m<=l; ++m) {
				matrix_type R_m_lp1(R_matrix(m,l+1,acf));
				matrix_type sum3(R_m_lp1.shape());
				sum3 = 0;
				for (int n=1; n<=m-1; ++n) {
					sum3 += R_matrix(m,n,acf)*PHI(m,n,l-m+1);
				}
				PHI(m+1,m,l-m).reference(Theta(m)*matrix_type(R_m_lp1 - sum3));
				for (int a=1; a<=m-1; ++a) {
					PHI(m+1,a,l-m).reference(matrix_type(
						PHI(m,a,l-m+1) - PHI(m,a,0)*PHI(m+1,m,l-m)
					));
				}
			}
		}
		blitz::cycleArrays(Pi_l, Pi_lp1);
	}
	#undef PHI
	return result_array(Pi_lp1, max_order);
}

template arma::Array3D<ARMA_REAL_TYPE>
arma::solve_yule_walker(
	Array3D<ARMA_REAL_TYPE> acf,
	const ARMA_REAL_TYPE,
	const int max_order
);

