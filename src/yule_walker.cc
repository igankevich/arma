#include "yule_walker.hh"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

#ifndef NDEBUG
#include <iostream>
#endif

#include "chop.hh"
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

	/**
	Computes row vector \f$\boldsymbol{\rho}_{i,j,k;d} = \left(
		\rho_{i-d,j-d,k-d},
		\boldsymbol{\rho}_{i,j,k;d,1},
		\ldots,
		\boldsymbol{\rho}_{i,j,k;d,d}
	\right)\f$.
	*/
	template <class T>
	inline blitz::Array<T,1>
	rho_vector(int i, int j, int k, int d, const blitz::Array<T,3>& acf) {
		#define ACF(x,y,z) acf(std::abs(x), std::abs(y), std::abs(z))
		typedef blitz::Array<T,1> vector_type;
		using blitz::Range;
		vector_type rho_ijkd(vector_shape(d));
		T* data = rho_ijkd.data();
		*data++ = ACF(i-d, j-d, k-d);
		/**
		Compute the first element, then for \f$e=1,\ldots,d\f$
		compute sub-vectors \f$\rho_{i,j,k;d,e} = \left(
			\boldsymbol{\rho}_{i,j,k;d,e}^{(1)},
			\ldots,
			\boldsymbol{\rho}_{i,j,k;d,e}^{(6)}
		\right)\f$ where
		*/
		for (int e=1; e<=d; ++e) {
			/**
			\f$\rho_{i,j,k;d,e}^{(1)} = \left(
				\rho_{i-d+e,j-d,k-d},
				\ldots,
				\rho_{i-d+e,j-d+e-1,k-d}
			\right)\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				*data++ = ACF(i-d+e, j-d+idx, k-d);
			}
			/**
			\f$\rho_{i,j,k;d,e}^{(2)} = \left(
				\rho_{i-d+e,j-d+e,k-d},
				\ldots,
				\rho_{i-d+1,j-d+e,k-d}
			\right)\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				*data++ = ACF(i-d+e-idx, j-d+e, k-d);
			}
			/**
			\f$\rho_{i,j,k;d,e}^{(3)} = \left(
				\rho_{i-d,j-d+e,k-d},
				\ldots,
				\rho_{i-d,j-d+e,k-d+e-1}
			\right)\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				*data++ = ACF(i-d, j-d+e, k-d+idx);
			}
			/**
			\f$\rho_{i,j,k;d,e}^{(4)} = \left(
				\rho_{i-d,j-d+e,k-d+e},
				\ldots,
				\rho_{i-d,j-d+1,k-d+e}
			\right)\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				*data++ = ACF(i-d, j-d+e-idx, k-d+e);
			}
			/**
			\f$\rho_{i,j,k;d,e}^{(5)} = \left(
				\rho_{i-d,j-d,k-d+e},
				\ldots,
				\rho_{i-d+e-1,j-d,k-d+e}
			\right)\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				*data++ = ACF(i-d+idx, j-d, k-d+e);
			}
			/**
			\f$\rho_{i,j,k;d,e}^{(5)} = \left(
				\rho_{i-d+e,j-d,k-d+e},
				\ldots,
				\rho_{i-d+e,j-d,k-d+1}
			\right)\f$.
			*/
			for (int idx=0; idx<e; ++idx) {
				*data++ = ACF(i-d+e, j-d, k-d+e-idx);
			}
		}
		#undef ACF
		return rho_ijkd;
	}

	/**
	Computes row vector for \f$d=0\f$ as
	\f$\boldsymbol{\rho}_{i,j,k;0} = \left( \rho_{i,j,k} \right)\f$.
	*/
	template <class T>
	inline T
	rho_scalar(int i, int j, int k, const blitz::Array<T,3>& acf) {
		#define ACF(x,y,z) acf(std::abs(x), std::abs(y), std::abs(z))
		return ACF(i, j, k);
		#undef ACF
	}

	/**
	Computes a part of Yule---Walker matrix \f$R_{a,b} =
	\begin{bmatrix}
	\boldsymbol{\rho}_{a,a,a;b}\\
	R_{a,b;1}\\
	\vdots\\
	R_{a,b;a}\\
	\end{bmatrix}\f$.
	*/
	template <class T>
	inline blitz::Array<T,2>
	R_matrix(int a, int b, const blitz::Array<T,3>& acf) {
		using blitz::Range;
		typedef blitz::Array<T,2> matrix_type;
		matrix_type R_ab(matrix_shape(a, b));
		R_ab(0, Range::all()) = rho_vector(a, a, a, b, acf);
		/**
		Compute the first row, then for \f$e=1,\ldots,a\f$
		compute submatrices \f$R_{a,b;e} = \begin{bmatrix}
		R_{a,b;e}^{(1)}\\
		\vdots\\
		R_{a,b;e}^{(6)}\\
		\end{bmatrix}\f$ where
		*/
		int offset = 1;
		for (int e=1; e<=a; ++e) {
			/**
			\f$R_{a,b;e}^{(1)} = \begin{bmatrix}
			\boldsymbol{\rho}_{a-e,a,a;b}\\
			\vdots\\
			\boldsymbol{\rho}_{a-e,a-e+1,a;b}\\
			\end{bmatrix}\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				R_ab(offset++, Range::all()) = rho_vector(a-e, a-idx, a, b, acf);
			}
			/**
			\f$R_{a,b;e}^{(2)} = \begin{bmatrix}
			\boldsymbol{\rho}_{a-e,a-e,a;b}\\
			\vdots\\
			\boldsymbol{\rho}_{a-1,a-e,a;b}\\
			\end{bmatrix}\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				R_ab(offset++, Range::all()) = rho_vector(a-e+idx, a-e, a, b, acf);
			}
			/**
			\f$R_{a,b;e}^{(3)} = \begin{bmatrix}
			\boldsymbol{\rho}_{a,a-e,a;b}\\
			\vdots\\
			\boldsymbol{\rho}_{a,a-e,a-e+1;b}\\
			\end{bmatrix}\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				R_ab(offset++, Range::all()) = rho_vector(a, a-e, a-idx, b, acf);
			}
			/**
			\f$R_{a,b;e}^{(4)} = \begin{bmatrix}
			\boldsymbol{\rho}_{a,a-e,a-e;b}\\
			\vdots\\
			\boldsymbol{\rho}_{a,a-1,a-e;b}\\
			\end{bmatrix}\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				R_ab(offset++, Range::all()) = rho_vector(a, a-e+idx, a-e, b, acf);
			}
			/**
			\f$R_{a,b;e}^{(5)} = \begin{bmatrix}
			\boldsymbol{\rho}_{a,a,a-e;b}\\
			\vdots\\
			\boldsymbol{\rho}_{a-e+1,a,a-e;b}\\
			\end{bmatrix}\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				R_ab(offset++, Range::all()) = rho_vector(a-idx, a, a-e, b, acf);
			}
			/**
			\f$R_{a,b;e}^{(6)} = \begin{bmatrix}
			\boldsymbol{\rho}_{a-e,a,a-e;b}\\
			\vdots\\
			\boldsymbol{\rho}_{a-e,a,a-1;b}\\
			\end{bmatrix}\f$.
			*/
			for (int idx=0; idx<e; ++idx) {
				R_ab(offset++, Range::all()) = rho_vector(a-e, a, a-e+idx, b, acf);
			}
		}
		return R_ab;
	}

	/**
	Computes a part of Yule---Walker matrix for \f$b=0\f$.
	*/
	template <class T>
	inline blitz::Array<T,1>
	R_vector_b0(int a, const blitz::Array<T,3>& acf) {
		using blitz::Range;
		using blitz::shape;
		typedef blitz::Array<T,1> vector_type;
		vector_type R_ab(vector_shape(a));
		T* data = R_ab.data();
		/**
		Compute the first row, then for \f$e=1,\ldots,a\f$
		compute submatrices \f$R_{a,0;e} = \begin{bmatrix}
		R_{a,0;e}^{(1)}\\
		\vdots\\
		R_{a,0;e}^{(6)}\\
		\end{bmatrix}\f$ where
		*/
		*data++ = rho_scalar(a, a, a, acf);
		for (int e=1; e<=a; ++e) {
			/**
			\f$R_{a,0;e}^{(1)} = \begin{bmatrix}
			\rho_{a-e,a,a}\\
			\vdots\\
			\rho_{a-e,a-e+1,a}\\
			\end{bmatrix}\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				*data++ = rho_scalar(a-e, a-idx, a, acf);
			}
			/**
			\f$R_{a,0;e}^{(2)} = \begin{bmatrix}
			\rho_{a-e,a-e,a}\\
			\vdots\\
			\rho_{a-1,a-e,a}\\
			\end{bmatrix}\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				*data++ = rho_scalar(a-e+idx, a-e, a, acf);
			}
			/**
			\f$R_{a,0;e}^{(3)} = \begin{bmatrix}
			\rho_{a,a-e,a}\\
			\vdots\\
			\rho_{a,a-e,a-e+1}\\
			\end{bmatrix}\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				*data++ = rho_scalar(a, a-e, a-idx, acf);
			}
			/**
			\f$R_{a,0;e}^{(4)} = \begin{bmatrix}
			\rho_{a,a-e,a-e}\\
			\vdots\\
			\rho_{a,a-1,a-e}\\
			\end{bmatrix}\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				*data++ = rho_scalar(a, a-e+idx, a-e, acf);
			}
			/**
			\f$R_{a,0;e}^{(5)} = \begin{bmatrix}
			\rho_{a,a,a-e}\\
			\vdots\\
			\rho_{a-e+1,a,a-e}\\
			\end{bmatrix}\f$,
			*/
			for (int idx=0; idx<e; ++idx) {
				*data++ = rho_scalar(a-idx, a, a-e, acf);
			}
			/**
			\f$R_{a,0;e}^{(6)} = \begin{bmatrix}
			\rho_{a-e,a,a-e}\\
			\vdots\\
			\rho_{a-e,a,a-1}\\
			\end{bmatrix}\f$.
			*/
			for (int idx=0; idx<e; ++idx) {
				*data++ = rho_scalar(a-e, a, a-e+idx, acf);
			}
		}
		return R_ab;
	}

	/**
	Computes a part of Yule---Walker matrix for \f$a=0\f$:
	\f$R_{0,b} = \boldsymbol{\rho}_{0,0,0;b} \f$.
	*/
	template <class T>
	inline blitz::Array<T,1>
	R_vector_a0(int b, blitz::Array<T,3> acf) {
		return rho_vector(0, 0, 0, b, acf);
	}

	/// Map result vector back to 3-dimensional array.
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

	template <class T, int N>
	inline bool
	is_square(const blitz::Array<T,N>& rhs) {
		const int first = rhs.extent(0);
		bool result = true;
		for (int i=1; i<rhs.dimensions(); ++i) {
			if (rhs.extent(i) != first) {
				result = false;
				break;
			}
		}
		return result;
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
arma::Yule_walker_solver<T>
::Yule_walker_solver(array_type acf, const T variance):
_acf(acf),
_variance(variance),
_maxorder(blitz::max(acf.shape()-1)) {
	if (!is_square(acf)) {
		throw std::invalid_argument("ACF is not square");
	}
}

template <class T>
arma::Yule_walker_solver<T>
::Yule_walker_solver(array_type acf):
Yule_walker_solver(array_type(acf/acf(0,0,0)), acf(0,0,0))
{}

template <class T>
void
arma::Yule_walker_solver<T>
::max_order(int rhs) {
	this->_maxorder = std::min(rhs, blitz::max(this->_acf.shape())-1);
}

template <class T>
typename arma::Yule_walker_solver<T>::array_type
arma::Yule_walker_solver<T>
::solve() {
	#define PHI(i,j,k) Phi[std::make_tuple(i,j,k)]
	typedef blitz::Array<T,2> matrix_type;
	typedef blitz::Array<T,1> vector_type;
	typedef std::tuple<int,int,int> key_type;
	typedef std::unordered_map<key_type,matrix_type> map3d_type;
	using blitz::Range;
	using blitz::shape;
	using blitz::any;
	/// Initial stage.
	const int max_order = this->_maxorder;
	map3d_type Phi;
	blitz::Array<vector_type,1> Pi_l(shape(max_order+2));
	blitz::Array<vector_type,1> Pi_lp1(shape(max_order+2));
	blitz::Array<matrix_type,1> Theta(shape(max_order+2));
	matrix_type R_1_1 = R_matrix(1, 1, this->_acf);
	vector_type R_1_0 = R_vector_b0(1, this->_acf);
	vector_type R_0_1 = R_vector_a0(1, this->_acf);
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
	T var0 = this->_variance;
	T var = this->_variance*lambda;
//	#ifndef NDEBUG
	/// Print solver state.
	std::clog << __func__ << ':' << "order=" << 1
			  << ",var=" << var << std::endl;
//	#endif
	Pi_l(1).reference(Pi_2_1);
	PHI(2,1,0).reference(matrix_type(R_sup_1_1*R_matrix(1, 2, this->_acf)));
	bool changed = false;
	int l = 1;
	do {
		++l;
		var0 = var;
		matrix_type R_l_l = R_matrix(l, l, this->_acf);
		vector_type R_l_0 = R_vector_b0(l, this->_acf);
		matrix_type sum1(R_l_l.shape());
		sum1 = 0;
		vector_type sum2(R_l_0.shape());
		sum2 = 0;
		for (int m=1; m<=l-1; ++m) {
			matrix_type R_l_m(R_matrix(l, m, this->_acf));
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
		var = this->_variance*lambda;
		if (l < max_order) {
			PHI(2,1,l-1).reference(R_sup_1_1*R_matrix(1, l+1, this->_acf));
			for (int m=2; m<=l; ++m) {
				matrix_type R_m_lp1(R_matrix(m,l+1,this->_acf));
				matrix_type sum3(R_m_lp1.shape());
				sum3 = 0;
				for (int n=1; n<=m-1; ++n) {
					sum3 += R_matrix(m,n,this->_acf)*PHI(m,n,l-m+1);
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
//		#ifndef NDEBUG
		/// Print solver state.
		std::clog << __func__ << ':' << "order=" << l
				  << ",var=" << var << std::endl;
//		#endif
		changed = !this->variance_has_not_changed_much(var, var0);
	} while (l < max_order && changed);
	#undef PHI
	array_type result;
	if (changed) {
		this->_varwn = var;
		result.reference(result_array(Pi_l, l));
	} else {
		this->_varwn = var0;
		result.reference(result_array(Pi_lp1, l));
	}
	if (this->_chop) {
		result.resizeAndPreserve(chop_right(result, this->_chopepsilon));
	}
	return result;
}

template class arma::Yule_walker_solver<ARMA_REAL_TYPE>;
