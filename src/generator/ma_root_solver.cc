#include "ma_root_solver.hh"

#include <algorithm>
#include <iostream>

namespace {

	template <class T>
	struct MA_params {
		arma::Array3D<T> _acf;
	};

	template <class T>
	int
	ma_coefficients_equation(
		const gsl_vector* theta,
		void* params,
		gsl_vector* result
	) {
		#define OFFSET(i,j,k) ((i)*nj*nk + (j)*nk + (k))
		MA_params<T>* p = reinterpret_cast<MA_params<T>*>(params);
		const arma::Array3D<T>& acf = p->_acf;
		const arma::Shape3D& shp = acf.shape();
		const int n = theta->size;
		const int ni = shp(0);
		const int nj = shp(1);
		const int nk = shp(2);
		T denominator = 0;
		for (int i=0; i<n; ++i) {
			const T theta_i = gsl_vector_get(theta, i);
			denominator += theta_i*theta_i;
		}
		for (int i=0; i<ni; ++i) {
			for (int j=0; j<nj; ++j) {
				for (int k=0; k<nk; ++k) {
					T numerator = -acf(i,j,k);
					for (int i1=i; i1<ni; ++i1) {
						for (int j1=j; j1<nj; ++j1) {
							for (int k1=k; k1<nk; ++k1) {
								numerator +=
									gsl_vector_get(theta, OFFSET(i1,j1,k1)) *
									gsl_vector_get(theta, OFFSET(i1-i,j1-j,k1-k));
							}
						}
					}
					gsl_vector_set(
						result,
						OFFSET(i,j,k),
						numerator / denominator
					);
				}
			}
		}
		return GSL_SUCCESS;
		#undef OFFSET
	}

}

template <class T>
arma::generator::MA_root_solver<T>
::MA_root_solver(Array3D<T> acf, bool use_derivatives):
_acf(acf / acf(0,0,0)),
_usederivatives(use_derivatives) {
	const int n = this->_acf.numElements();
	if (this->_usederivatives) {
		this->_fdfsolver =
			gsl_multiroot_fdfsolver_alloc(gsl_multiroot_fdfsolver_gnewton, n);
	} else {
		this->_fsolver =
			gsl_multiroot_fsolver_alloc(gsl_multiroot_fsolver_dnewton, n);
	}
}

template <class T>
arma::generator::MA_root_solver<T>
::~MA_root_solver() {
	if (this->_usederivatives) {
		gsl_multiroot_fdfsolver_free(this->_fdfsolver);
	} else {
		gsl_multiroot_fsolver_free(this->_fsolver);
	}
}

template <class T>
arma::Array3D<T>
arma::generator::MA_root_solver<T>
::solve() {
	return this->_usederivatives
		? this->solve_with_derivatives()
		: this->solve_without_derivatives();
}

template <class T>
arma::Array3D<T>
arma::generator::MA_root_solver<T>
::solve_without_derivatives() {
	/// Initialise GSL multi-root solver.
	const int n = this->_acf.numElements();
	gsl_vector* theta = gsl_vector_alloc(n);
	gsl_vector_set_zero(theta);
	gsl_vector_set(theta, 0, T(-1));
	MA_params<T> params {this->_acf};
	gsl_multiroot_function f {&ma_coefficients_equation<T>, size_t(n), &params};
	gsl_multiroot_fsolver_set(this->_fsolver, &f, theta);
	const int max_iter = this->_maxiterations;
	const T max_res = this->_maxresidual;
	int iter = 0;
	int status = 0;
	do {
		++iter;
		std::clog << __func__ << ": iteration=" << iter << std::endl;
		status = gsl_multiroot_fsolver_iterate(this->_fsolver);
		if (status) {
			std::clog << __func__ << ": the solver is stuck" << std::endl;
			break;
		}
		/// Calculate residual.
		Array3D<T> residual(this->_acf.shape());
		std::copy_n(this->_fsolver->f->data, n, residual.data());
		Array3D<T> tmp(this->_acf.shape());
		std::copy_n(this->_fsolver->x->data, n, tmp.data());
		const T res = blitz::max(blitz::abs(residual));
		std::clog << __func__ << ": iteration=" << iter
			<< ",res=" << res
			<< ",theta=" << tmp
			<< ",residual=" << residual
			<< std::endl;
		status = gsl_multiroot_test_residual(this->_fsolver->f, max_res);
	} while (status == GSL_CONTINUE && iter < max_iter);
	/// Copy-out the result.
	Array3D<T> result(this->_acf.shape());
	std::copy_n(this->_fsolver->x->data, n, result.data());
	gsl_vector_free(theta);
	return result;
}

template <class T>
arma::Array3D<T>
arma::generator::MA_root_solver<T>
::solve_with_derivatives() {
	throw std::runtime_error("not implemented");
}

template class arma::generator::MA_root_solver<ARMA_REAL_TYPE>;
