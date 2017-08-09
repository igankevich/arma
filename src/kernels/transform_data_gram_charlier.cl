#include "common.h"

#if arma_float
#define SQRT2 1.4142135623730950f
#define SQRT2PI 2.5066282746310005f
#else
#define SQRT2 1.4142135623730950
#define SQRT2PI 2.5066282746310005
#endif

inline T
gaussian_cdf(const T x, const T mean, const T stdev) {
	#if arma_float
	return 0.5f*(1.0f + erf((x - mean) / (SQRT2 * stdev)));
	#else
	return 0.5*(1.0 + erf((x - mean) / (SQRT2 * stdev)));
	#endif
}

inline T
gram_charlier_cdf(const T x, const T skewness, const T kurtosis) {
	#if arma_float
	return exp(-0.5f*x*x)*(kurtosis*(3.0f*x - x*x*x)
		+ skewness*(4.0f - 4.0f*x*x) + 3.0f*x*x*x - 9.0f*x)
		/ (24.0f*SQRT2PI) + 0.5f*erf(x/SQRT2) + 0.5f;
	#else
	return exp(-0.5*x*x)*(kurtosis*(3.0*x - x*x*x)
		+ skewness*(4.0 - 4.0*x*x) + 3.0*x*x*x - 9.0*x)
		/ (24.0*SQRT2PI) + 0.5*erf(x/SQRT2) + 0.5;
	#endif
}

inline T
equation_cdf(
	const T x,
	const T skewness,
	const T kurtosis,
	const T gaussian_cdf_data
) {
	return gram_charlier_cdf(x, skewness, kurtosis) - gaussian_cdf_data;
}

inline T
bisection_equation_cdf_gram_charlier(
	T a,
	T b,
	const int niterations,
	const T gaussian_cdf_data,
	const T skewness,
	const T kurtosis
) {
	T c, fc;
	for (int i=0; i<niterations; ++i) {
		#if arma_float
		c = 0.5f * (a + b);
		#else
		c = 0.5 * (a + b);
		#endif
		fc = equation_cdf(c, skewness, kurtosis, gaussian_cdf_data);
		if (equation_cdf(a, skewness, kurtosis, gaussian_cdf_data) * fc < 0) b = c;
		if (equation_cdf(b, skewness, kurtosis, gaussian_cdf_data) * fc < 0) a = c;
	}
	return c;
}

kernel void
transform_data_gram_charlier(
	global T* data,
	const T a,
	const T b,
	const int niterations,
	const T stdev,
	const T skewness,
	const T kurtosis
) {
	const int nz = get_global_size(0);
	const int nx = get_global_size(1);
	const int ny = get_global_size(2);
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	const int k = get_global_id(2);
	const int offset = i*nx*ny + j*ny + k;
	data[offset] = bisection_equation_cdf_gram_charlier(
		a,
		b,
		niterations,
		gaussian_cdf(data[offset], 0, stdev),
		skewness,
		kurtosis
	);
}
