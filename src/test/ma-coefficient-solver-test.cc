#include <gtest/gtest.h>

#include <algorithm>
#include <random>

#include "generator/ma_coefficient_solver.hh"
#include "blitz.hh"
#include "arma.hh"
#include "domain.hh"

#include <dlib/global_optimization.h>

typedef ARMA_REAL_TYPE T;

template <class X>
using column_vector = dlib::matrix<X,0,1>;

/*
TEST(MACoefficientSolver, Simple) {
	using namespace arma;
	using blitz::RectDomain;
	using blitz::shape;
	using blitz::sum;
	using blitz::pow2;
	using std::abs;
	Array3D<T> acf(shape(4,1,1));
	acf(0,0,0) = T(4);
	acf(1,0,0) = T(-0.5);
	acf(2,0,0) = T(0.25);
	acf(3,0,0) = T(-0.1);
	const int n = acf.numElements();
	const int m = n-1;
	auto objective = [n,m,&acf] (const column_vector<T>& theta_in) {
		using blitz::sum;
		using blitz::pow2;
		using blitz::mean;
		using std::sqrt;
		const Shape3D& shp = acf.shape();
		const int ni = shp(0);
		const int nj = shp(1);
		const int nk = shp(2);
		Array3D<T> theta(shp);
		theta(0,0,0) = T(-1);
		for (int i=0; i<m; ++i) {
			*(theta.data() + i + 1) = theta_in(i);
		}
		const T var_wn = theta_in(m);
//		const T denominator = sum(pow2(theta));
		Array3D<T> residual(shp);
		for (int i=0; i<ni; ++i) {
			for (int j=0; j<nj; ++j) {
				for (int k=0; k<nk; ++k) {
					RectDomain<3> sub1(Shape3D(i, j, k), shp - 1);
					RectDomain<3> sub2(Shape3D(0, 0, 0),
									   shp - Shape3D(i, j, k) - 1);
					T numerator = blitz::sum(theta(sub1) * theta(sub2));
					T elem = std::abs(numerator*var_wn - acf(i,j,k));
					residual(i,j,k) = elem;
				}
			}
		}
		const T last_residual = std::abs(sum(pow2(theta))*var_wn - acf(0,0,0));
		const T max_residual = sqrt(mean(pow2(residual) + pow2(last_residual)));
		std::clog << "max_residual=" << max_residual
			<< ",theta(0,0,0)=" << theta(0,0,0)
			<< ",acf(0,0,0)=" << acf(0,0,0)
			<< ",residual(0,0,0)=" << residual(0,0,0)
			<< ",var_wn=" << var_wn
			<< std::endl;
		return max_residual;
	};
	column_vector<T> theta_min(m+1);
	column_vector<T> theta_max(m+1);
	theta_min = T(-100);
	theta_max = T(100);
//	theta_min(m) = T(0);
//	theta_max(m) = T(0.1);//this->_acf(0,0,0);
	auto res = dlib::find_min_global(
		objective,
		theta_min,
		theta_max,
		dlib::max_function_calls(400)
	);
	Array3D<T> result(acf.shape());
	result(0,0,0) = -1;
	for (int i=0; i<m; ++i) {
		*(result.data() + i + 1) = res.x(i);
	}
	const T var_wn = res.x(m);
	std::clog << "result=" << result << std::endl;
	std::clog << "var_wn=" << var_wn << std::endl;
	std::clog << "residual=" << abs(sum(pow2(result))*var_wn - acf(0,0,0)) << std::endl;
	arma::validate_process(result);
}
*/

template <class T>
arma::Array3D<T>
ma_coefficients_bisection(arma::Array3D<T> acf, T var_wn_0) {
	using namespace arma;
	const Shape3D& order = acf.shape();
	Array3D<T> a(order);
	Array3D<T> b(order);
	Array3D<T> c(order);
	Array3D<T> fc(order);
	Array3D<T> fa(order);
	Array3D<T> fb(order);
	std::default_random_engine rng;
	std::uniform_real_distribution<T> dist(T(-1), T(1));
	std::generate_n(a.data(), a.numElements(), std::bind(dist, rng));
	std::generate_n(b.data(), b.numElements(), std::bind(dist, rng));
	a += T(-2);
	b += T(2);
	a(0,0,0) = -1;
	b(0,0,0) = -1;
	auto func = [&acf] (const Array3D<T>& theta, T var_wn) {
		using blitz::RectDomain;
		using blitz::sum;
		using blitz::pow2;
		using blitz::mean;
		const Shape3D& shp = theta.shape();
		const int ni = shp(0);
		const int nj = shp(1);
		const int nk = shp(2);
//		const T denominator = sum(pow2(theta));
//		const T var_wn = 0.1;
		Array3D<T> residual(shp);
		for (int i=0; i<ni; ++i) {
			for (int j=0; j<nj; ++j) {
				for (int k=0; k<nk; ++k) {
					RectDomain<3> sub1(Shape3D(i, j, k), shp - 1);
					RectDomain<3> sub2(Shape3D(0, 0, 0),
									   shp - Shape3D(i, j, k) - 1);
					T numerator = sum(theta(sub1) * theta(sub2));
					T elem = numerator*var_wn - acf(i,j,k);
					residual(i,j,k) = elem;
				}
			}
		}
//		const T max_residual = blitz::max(abs(residual));
//		std::clog << "max_residual=" << max_residual
//			<< ",theta(0,0,0)=" << theta(0,0,0)
//			<< ",acf(0,0,0)=" << acf(0,0,0)
//			<< ",residual(0,0,0)=" << residual(0,0,0)
////			<< ",theta=" << theta
//			<< std::endl;
		return residual;
	};
	T max_residual_0 = std::numeric_limits<T>::min();
	T max_residual = std::numeric_limits<T>::max();
	for (int i=0; i<100 && std::abs(max_residual - max_residual_0) > T(1e-5); ++i) {
		c = T(0.5)*(a + b);
		//c(0,0,0) = -1;
		fc = func(c, var_wn_0);
		fa = func(a, var_wn_0);
		fb = func(b, var_wn_0);
		b = blitz::where(fa*fc < T(0), c, b);
		a = blitz::where(fb*fc < T(0), c, a);
		max_residual_0 = max_residual;
		max_residual = max(abs(fc));
		var_wn_0 = MA_white_noise_variance(acf, c);
		std::clog << __func__
			<< "iteration=" << i
			<< ",residual=" << max_residual
			<< ",varwn=" << var_wn_0
			<< std::endl;
//		std::clog << __func__ << ": "
//			<< "fa=" << fa
//			<< ",fc=" << fc
//			<< ",fb=" << fb
//			<< std::endl;
//		std::clog << __func__ << ": "
//			<< ",a=" << a
//			<< ",c=" << c
//			<< ",b=" << b
//			<< std::endl;
	}
	return c;
}

TEST(MACoefficientSolver, Bisection) {
	using namespace arma;
	using blitz::sum;
	using blitz::pow2;
//	const Shape3D order(4,1,1);
//	Array3D<T> acf(order);
//	acf(0,0,0) = T(1.99114);
//	acf(1,0,0) = T(-0.13142);
//	acf(2,0,0) = T(0.41992);
//	acf(3,0,0) = T(0.118226);
	Array3D<T> acf;
	{ std::ifstream("acf333") >> acf; }
	std::ofstream out("var_wn_data");
	const int n = 100;
	Domain<T,1> grid{{T(0)}, {T(acf(0,0,0)*2)}, {n}};
	Array3D<T> theta(acf.shape());
	for (int j=0; j<n; ++j) {
		const T var_wn_0 = grid(j,0);
//		for (int i=0; i<10; ++i) {
			theta = ma_coefficients_bisection(acf, var_wn_0);
			T var_wn = MA_white_noise_variance(acf, theta);
			std::clog << "var_wn=" << var_wn << std::endl;
			const T d = std::abs(sum(pow2(theta))*var_wn_0 - acf(0,0,0));
			out << var_wn_0 << ' ' << d << ' ' << var_wn << std::endl;
//		}
	}
//	auto objective = [&theta,&acf] (T var_wn) {
//		using blitz::sum;
//		using blitz::pow2;
//		theta = ma_coefficients_bisection(acf, var_wn);
//		var_wn = MA_white_noise_variance(acf, theta);
//		return std::abs(sum(pow2(theta))*var_wn - acf(0,0,0));
//	};
//	auto res = dlib::find_min_global(
//		objective,
//		T(0),
//		T(10),
//		dlib::max_function_calls(400)
//	);
//	std::clog << "var_wn=" << res.x << std::endl;
	std::clog << "theta=" << theta << std::endl;
	arma::validate_process(theta);
}
