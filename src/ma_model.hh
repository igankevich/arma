#ifndef MA_MODEL_HH
#define MA_MODEL_HH

#include <cassert>   // for assert
#include <algorithm> // for copy_n

#include "types.hh" // for size3, ACF, AR_coefs, Zeta, Array2D

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h> // for gsl_multiroot_fsolver_iterate

namespace autoreg {

	std::ostream& operator<<(std::ostream& out, const gsl_vector* rhs) {
		out << "[ ";
		for (size_t i = 0; i < rhs->size; ++i) {
			out << gsl_vector_get(rhs, i) << ' ';
		}
		out << ']';
		return out;
	}

	template <class T>
	struct equation_params {
		const ACF<T>& acf;
		size3 order;
	};

	template <class T>
	struct Moving_average_model {

		Moving_average_model(ACF<T> acf, size3 order)
		    : _acf(acf / acf(0, 0, 0)), _order(order),
		      _theta(compute_MA_coefs()) {}

		T
		white_noise_variance() {
			return _acf(0, 0, 0) / (T(1) + blitz::sum(blitz::pow2(_theta)));
		}

		Zeta<T> operator()(Zeta<T> eps) {
			Zeta<T> zeta(eps.shape());
			const size3 fsize = _theta.shape();
			const size3 zsize = zeta.shape();
			const int t1 = zsize[0];
			const int x1 = zsize[1];
			const int y1 = zsize[2];
			for (int t = 0; t < t1; t++) {
				for (int x = 0; x < x1; x++) {
					for (int y = 0; y < y1; y++) {
						const int m1 = std::min(t + 1, fsize[0]);
						const int m2 = std::min(x + 1, fsize[1]);
						const int m3 = std::min(y + 1, fsize[2]);
						T sum = 0;
						for (int k = 0; k < m1; k++)
							for (int i = 0; i < m2; i++)
								for (int j = 0; j < m3; j++)
									sum += _theta(k, i, j) *
									       eps(t - k, x - i, y - j);
						zeta(t, x, y) = sum;
					}
				}
			}
			return zeta;
		}

		/// TODO
		void
		validate() {}

	private:
		static int
		moving_average_equation(const gsl_vector* in_x, void* params,
		                        gsl_vector* out_f) {
			std::clog << "eq\n";
			const equation_params<T>* par =
			    static_cast<const equation_params<T>*>(params);
			const ACF<T>& acf = par->acf;
			const size3& order = par->order;
			// copy in
			Array3D<T> x(order);
			x(0, 0, 0) = T(-1);
			std::copy_n(in_x->data, in_x->size, x.data() + 1);
			const T denominator = blitz::sum(blitz::pow2(x));
			Array1D<T> f(x.size());
			int idx = 0;
			for (int i = 0; i < order(0); ++i) {
				for (int j = 0; j < order(1); ++j) {
					for (int k = 0; k < order(2); ++k) {
						T sum = 0;
						for (int l = i; l < order(0); ++l) {
							for (int m = j; m < order(1); ++m) {
								for (int n = k; n < order(2); ++n) {
									sum += x(l, m, n) * x(l - i, m - j, n - k);
								}
							}
						}
						f(idx) = sum / denominator - acf(i, j, k);
						++idx;
					}
				}
			}
			// copy out
			std::copy_n(f.data() + 1, out_f->size, out_f->data);
			return GSL_SUCCESS;
		}

		void
		print_state(size_t iter, gsl_multiroot_fsolver* s) {
			std::clog << "iteration=" << iter << ", x=" << s->x
			          << ", f(x)=" << s->f << std::endl;
		}

		/// Solve nonlinear system to find moving-average coefficients.
		AR_coefs<T>
		compute_MA_coefs() {
			/// No. of equations equals MA model order.
			const size_t n = blitz::product(_order) - 1;
			assert(n > 0);
			equation_params<T> params = {_acf, _order};
			gsl_multiroot_function f = {&moving_average_equation, n, &params};
			gsl_vector* x = gsl_vector_calloc(n);
			std::fill_n(x->data, 1, T(-0.5));
			const gsl_multiroot_fsolver_type* type =
			    gsl_multiroot_fsolver_hybrids;
			gsl_multiroot_fsolver* solver =
			    gsl_multiroot_fsolver_alloc(type, n);
			gsl_multiroot_fsolver_set(solver, &f, x);
			int iter = 0;
			int status = 0;
			do {
				++iter;
				status = gsl_multiroot_fsolver_iterate(solver);

				print_state(iter, solver);

				if (status) /* check if solver is stuck */
					break;

				status = gsl_multiroot_test_residual(solver->f, 1e-7);
			} while (status == GSL_CONTINUE && iter < 10);
			std::clog << "status = " << gsl_strerror(status) << std::endl;
			// copy out
			AR_coefs<T> coefs(_order);
			coefs(0) = 0;
			std::copy_n(solver->x->data, solver->x->size, coefs.data() + 1);
			gsl_multiroot_fsolver_free(solver);
			gsl_vector_free(x);
			return coefs;
		}

		ACF<T> _acf;
		size3 _order;
		AR_coefs<T> _theta;
	};
}

#endif // MA_MODEL_HH
