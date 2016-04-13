#ifndef AUTOREG_HH
#define AUTOREG_HH

#include <random>
#include <iterator>
#include <functional>
#include <sstream>

#include <blitz/array.h>

#include "stdx.hh"
#include "vector_n.hh"

/// @file
/// File with subroutines for AR model, Yule-Walker equations
/// and some others.

namespace autoreg {

	template<class T> using ACF = blitz::Array<T,3>;
	template<class T> using AR_coefs = blitz::Array<T,3>;
	template<class T> using Matrix = blitz::Array<T,2>;

	template<class T>
	ACF<T> approx_acf(T alpha, T beta, T gamm, const Vec3<T>& delta, const size3& acf_size) {
		ACF<T> acf(acf_size);
		blitz::firstIndex t;
		blitz::secondIndex x;
		blitz::thirdIndex y;
		acf = gamm
			* std::exp(-alpha * (t*delta[0] + x*delta[1] + y*delta[2]))
			* std::cos(beta*t*delta[0]
			* std::cos(beta*x*delta[1]);
			//*cos(beta*y*delta[2]);
		return acf;
	}

	template<class T>
	T white_noise_variance(const AR_coefs<T>& ar_coefs, const ACF<T>& acf) {
		return acf(0,0,0) - blitz::sum(ar_coefs * acf);
	}

	template<class T>
	T ACF_variance(const ACF<T>& acf) {
		return acf(0,0,0);
	}

	/// Удаление участков разгона из реализации.
	template<class T>
	void trim_zeta(const std::valarray<T>& zeta2,
	               const size3& zsize2,
	               const size3& zsize,
	               std::valarray<T>& zeta)
	{
		const size3 delta = zsize2-zsize;
		const size_t offset = blitz::product(delta);
		const std::gslice end_part{
			offset,
			{delta[0], delta[1], delta[2]},
			{1, 1, 1}
		};
		zeta = zeta2[end_part];
	}

	template<class T>
	bool is_stationary(AR_coefs<T>& phi) {
		return blitz::all(std::abs(phi) > T(1));
	}

	template<class T>
	AR_coefs<T>
	compute_AR_coefficients(const ACF& acf) {
		AR_coefs phi(acf.shape());
		phi = acf;
		const size_t m = phi.numElements();
		Matrix<T> lhs(blitz::shape(m, m));
		sysv<T>('U', m, 1, lhs.data(), m, phi.data(), m);
		if (!is_stationary(phi)) {
			throw std::runtime_error("AR process is not stationary (|phi| > 1)");
		}
		return phi;
	}

template<class T>
struct Yule_walker {

	Yule_walker(const std::valarray<T>& acf_, const size3& acf_size_, std::valarray<T>& a_, std::valarray<T>& b_):
	acf(acf_), acf_size(acf_size_), a(a_), b(b_)
	{}

	void act() {
		const int n = acf.size() - 1;
		for (int i=0; i<n; ++i) {
			const int n = acf.size()-1;
			const Index<3> id(acf_size);
			const Index<2> ida(size2(n, n));
			for (int j=0; j<n; j++) {
			    // casting to signed type ptrdiff_t
				int i2 = id(sub_abs(id.x(i+1), id.x(j+1)),
				            sub_abs(id.y(i+1), id.y(j+1)),
				            sub_abs(id.t(i+1), id.t(j+1)) );
				int i1 = i*n + j; //ida(i, j);
//				cout << "i  = " << i << endl;
//				cout << "j  = " << j << endl;
//				cout << "i2 = " << i2 << endl;
				a[i1] = acf[i2];
			}
		}
		const int m = b.size();
		for (int i=0; i<m; ++i) {
			const Index<3> id(acf_size);
			b[i] = acf[id( id.x(i+1), id.y(i+1), id.t(i+1) )];
		}
	}

private:
	int sub_abs(int a, int b) {
	    return (a > b) ? a-b : b-a;
	}

	const std::valarray<T>& acf;
	const size3& acf_size;
	std::valarray<T>& a;
	std::valarray<T>& b;
};

/*
template<class T>
struct Solve_Yule_Walker {

	Solve_Yule_Walker(std::valarray<T>& ar_coefs2, std::valarray<T>& aa, std::valarray<T>& bb, const size3& acf_size):
		ar_coefs(ar_coefs2), a(aa), b(bb), _acf_size(acf_size)
	{}

	void act() {

		int m = ar_coefs.size()-1;
		int info = 0;
		sysv<T>('U', m, 1, &a[0], m, &b[0], m, &info);
		if (info != 0) {
			std::stringstream s;
			s << "ssysv error, D(" << info << ", " << info << ")=0";
			throw std::invalid_argument(s.str());
		}

		std::copy(&b[0], &b[m], &ar_coefs[1]);
		ar_coefs[0] = 0;

		if (!is_stationary(ar_coefs)) {
			std::stringstream msg;
			msg << "Process is not stationary: |f(i)| > 1\n";
//			int n = ar_coefs.size();
			Index<3> idx(_acf_size);
			for (size_t i=0; i<_acf_size[0]; ++i)
				for (size_t j=0; j<_acf_size[1]; ++j)
					for (size_t k=0; k<_acf_size[2]; ++k)
						if (std::abs(ar_coefs[idx(i, j, k)]) > T(1))
							msg << "ar_coefs[" << i << ',' << j << ',' << k << "] = " << ar_coefs[idx(i, j, k)] << '\n';
			throw std::runtime_error(msg.str());
//			std::cerr << "Continue anyway? y/N\n";
//			char answer = 'n';
//			std::cin >> answer;
//			if (answer == 'n' || answer == 'N') {
//				throw std::runtime_error("Process is not stationary: |f[i]| >= 1.");
//			}
		}
	}

private:
	std::valarray<T>& ar_coefs;
	std::valarray<T>& a;
	std::valarray<T>& b;
	const size3& _acf_size;
};
*/

	/// Генерация белого шума по алгоритму Вихря Мерсенна и
	/// преобразование его к нормальному распределению по алгоритму Бокса-Мюллера.
	template<class T>
	void generate_white_noise(std::valarray<T>& eps, const T var_eps) {
		if (var_eps < T(0)) {
			throw std::runtime_error("variance is less than zero");
		}

		// инициализация генератора
		std::mt19937 generator;
		#if !defined(DISABLE_RANDOM_SEED)
		generator.seed(std::chrono::steady_clock::now().time_since_epoch().count());
		#endif
		std::normal_distribution<T> normal(T(0), var_eps);

		// генерация и проверка
		std::generate(std::begin(eps), std::end(eps), std::bind(normal, generator));
		if (std::any_of(std::begin(eps), std::end(eps), &stdx::isnan<T>)) {
			throw std::runtime_error("white noise generator produced some NaNs");
		}
	}

/// Генерация отдельных частей реализации волновой поверхности.
template<class T>
void generate_zeta(const std::valarray<T>& phi,
				   const size3& fsize,
				   const size3& zsize,
				   std::valarray<T>& zeta)
{
	const Index<3> id(fsize);
	const Index<3> idz(zsize);
	const std::size_t t1 = zsize[0];
	const std::size_t x1 = zsize[1];
	const std::size_t y1 = zsize[2];
    for (std::size_t t=0; t<t1; t++) {
        for (std::size_t x=0; x<x1; x++) {
            for (std::size_t y=0; y<y1; y++) {
                const std::size_t m1 = std::min(t+1, fsize[0]);
                const std::size_t m2 = std::min(x+1, fsize[1]);
                const std::size_t m3 = std::min(y+1, fsize[2]);
                T sum = 0;
                for (std::size_t k=0; k<m1; k++)
                    for (std::size_t i=0; i<m2; i++)
                        for (std::size_t j=0; j<m3; j++)
                            sum += phi[id(k, i, j)]*zeta[idz(t-k, x-i, y-j)];
                zeta[idz(t, x, y)] += sum;
            }
        }
    }
}

}

#endif // AUTOREG_HH
