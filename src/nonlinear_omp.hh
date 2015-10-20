#ifndef NONLINEAR_OMP_HH
#define NONLINEAR_OMP_HH

#include <gsl/gsl_math.h>
#include <gsl/gsl_cdf.h>
#include "emath.hh"
#include "valarray_ext.hh"

/// Implementations of nonlinear inertialess transform (NIT)
/// to capture asymmetry of z-coordinate distribution 
/// of ocean waves.
namespace autoreg {

const float EPS     = 1e-6;
const float MAX_ERR = 0.02;
const float SQRT2   = 1.4142135623730951f;
const float SQRT2PI = 2.5066282749176025f;

/// Skew normal distribution. Approximation with @skewness and @kurtosis.
template<class T>
struct Skew_normal {

	Skew_normal(T skew, T kurt):
	skewness(skew), kurtosis(kurt)
	{}

	T operator()(T x) {
		return std::exp(-0.5*x*x)*(kurtosis*(3*x - x*x*x)
		       + skewness*(4 - 4*x*x) + 3*x*x*x - 9*x)/(24*std::sqrt(2*M_PI))
			   + 0.5*std::erf(x/std::sqrt(2.0)) + 0.5;
	}

private:
	T skewness;
	T kurtosis;
};

/// Skew normal distribution. Approximation with coefficient @a
/// that governs both skewness and kurtosis.
template<class T>
class Skew_normal_2 {

	explicit Skew_normal_2(T c): alpha(c) {}

	T operator()(T x) const
	{
		return gsl_cdf_ugaussian(x) - T(2)*owenT(x, alpha);
	}

private:
	T alpha;
};

/// Weibull distribution. Mostly for fun, ocean waves have different shape :-)
template<class T>
struct Weibull {

	Weibull(T a, T b): alpha(a), beta(b) {}

	T operator()(T x) {
		return T(1) - exp(-pow(x/beta, alpha));
	}

private:
	T alpha;
	T beta;
};

/// Skew normal distribution polynomial.
template<class T, class CDF>
struct Equation_cdf {

	Equation_cdf(CDF func, T y): cdf(func), cdf_y(y) {}

	T operator()(T x) {
		return cdf(x) - cdf_y;
	}

private:
	CDF cdf;
	T cdf_y;
};

/// Forward nonlinear transform.
template<class T, class CDF>
void transform_norm_distribution(T x0,
								 T x1,
								 std::valarray<T>& arr_x,
								 std::valarray<T>& arr_y,
								 CDF cdf)
{
	const T DINTERVAL = 10;
	T dx = (x1-x0)/(arr_x.size()-1);
	for (size_t i=0; i<arr_x.size(); i++) {
		T x = x0 + i*dx;
		arr_x[i] = x;
		arr_y[i] = bisection<T>(x-DINTERVAL, x+DINTERVAL, Equation_cdf<T, CDF>(cdf, gsl_cdf_ugaussian_P(x)), EPS, 60);
	}
}

template<class T, class CDF>
void interpolation_coefs(T x0,
						 T x1,
						 size_t nodes,
						 std::valarray<T>& interp_coefs,
						 CDF cdf)
{
	std::valarray<T> node_x(nodes), node_y(nodes);  // interpolation grid
	transform_norm_distribution(x0, x1, node_x, node_y, cdf);
#ifdef DEBUG
	write("node_x", node_x);
	write("node_y", node_y);
#endif
	interpolate(&node_x[0], &node_y[0], node_x.size(), &interp_coefs[0], interp_coefs.size());
}

template<class T>
T poly(T x, std::valarray<T>& a)
{
	T sum = 0;
	T x2 = 1;
	for (size_t k=0; k<a.size(); k++) {
		sum += a[k]*x2;
		x2 *= x;
	}
	return sum;
}

template<class T>
class Equation_acf {
	std::valarray<T>& c;
	T acf_i;
public:
	Equation_acf(std::valarray<T>& coefs, T y): c(coefs), acf_i(y) {}
	T operator()(T x) const {
		T sum = 0;
		T f = 1;
		T x2 = 1;
		for (size_t i=0; i<c.size(); ++i) {
			sum += c[i]*c[i]*x2/f;
			f *= (i+1);
			x2 *= x;
		}
		return sum - acf_i;
	}
};

template<class T>
std::valarray<T> determine_nit_coefs(const std::valarray<T>& a,
								size_t max_coefs,
								const std::valarray<T>& acf)
{
	std::valarray<T> c(max_coefs);
	T sum_c = 0;
	T f = 1;
	T err = std::numeric_limits<T>::max();
	size_t trim = 0;
	for (size_t m=0; m<c.size(); ++m) {
		
		// calc coef
		Poly<T> y = Poly<T>(a) * hermite_poly<T>(m);
		T sum2 = y[0];
		for (size_t i=2; i<=y.order(); i+=2)
			sum2 += y[i]*fact<T>(i-1, 2);
		c[m] = sum2;

		// calc error
		sum_c += c[m]*c[m]/f;
		f *= (m+1);
		T e = abs(acf[0] - sum_c);
		// критерий возможно правильно писать так: abs(T(1) - sum_c)

		// determine minimum error
		if (e < err) {
			err = e;
			trim = m+1;
		}
		std::clog << "err = " << e << std::endl;
	}
	std::clog << "trim = " << trim << std::endl;
	return c[std::slice(0, trim, 1)];
}

template<class T>
void transform_acf(std::valarray<T>& acf, std::valarray<T>& c)
{
	//ifdebug("nit coefs trimmed =\n" << c << endl);
	const T ACF_INTERVAL = 2.0;
	for (size_t i=0; i<acf.size(); ++i)
		acf[i] = bisection<T>(-ACF_INTERVAL, ACF_INTERVAL, Equation_acf<T>(c, acf[i]), EPS, 30);
}

// нелинейное безынерционной преобразование ковариационной поверхности
template<class T>
void transform_acf(std::valarray<T>& interp_coefs,
				   size_t max_coefs,
				   std::valarray<T>& acf)
{
	std::valarray<T> c = determine_nit_coefs<T>(interp_coefs, max_coefs, acf);
#ifdef DEBUG
	clog << "nit coefs =\n" << c << endl;
#endif
	transform_acf<T>(acf, c);
}


// Функция преобразует аппликаты волновой поверхности к асимметричному распределению.
// Используется либо интерполяция, либо решения уравнения F(y) = Phi(x)
template<class T, class CDF>
void transform_water_surface(std::valarray<T>& a,
							 const size3& zsize,
							 std::valarray<T>& z,
							 CDF cdf, 
					         T min_z,
							 T max_z)
{
	const Index<3> id(zsize);
	for (size_t x=0; x<zsize[0]; x++) {
		for (size_t y=0; y<zsize[1]; y++) {
			for (size_t t=0; t<zsize[2]; t++) {
				T z0 = z[id(x, y, t)];
				if (z0 > min_z && z0 < max_z) {
					z[id(x, y, t)] = poly<T>(z0, a);
				} else {
					z[id(x, y, t)] = bisection<T>(min_z, max_z, Equation_cdf<T, CDF>(cdf, gsl_cdf_ugaussian_P(z0)), EPS, 30);
				}
			}
		}
	}
}

}

#endif // NONLINEAR_OMP_HH
