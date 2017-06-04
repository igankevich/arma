#include "owen_t.hh"
#include "physical_constants.hh"
#include <cmath>

namespace {

	const int nabscissas = 10;

	template <class T>
	constexpr const T abscissas[nabscissas] = {
		T(0.07652652113349734),
		T(0.22778585114164507),
		T(0.37370608871541955),
		T(0.5108670019508271),
		T(0.636053680726515),
		T(0.7463319064601508),
		T(0.8391169718222188),
		T(0.912234428251326),
		T(0.9639719272779138),
		T(0.9931285991850949)
	};

	template <class T>
	constexpr const T weights[nabscissas] = {
		T(0.1527533871307259),
		T(0.14917298647260419),
		T(0.1420961093183816),
		T(0.13168863844917506),
		T(0.1181945319620929),
		T(0.10193011981877752),
		T(0.08327674156219758),
		T(0.06267204832636279),
		T(0.040601429800683646),
		T(0.017614007137058983)
	};
}

template <class T>
T
arma::apmath::owen_t(T h, T alpha) {
	using arma::constants::_2pi;
	using std::exp;
	T result = 0;
	const T a2 = alpha*alpha;
	const T h2 = -T(0.5)*h*h;
	for (int i=0; i<nabscissas; ++i) {
		const T x = abscissas<T>[i];
		const T term = (T(1) + a2*x*x);
		result += weights<T>[i] * exp(h2*term) / term;
	}
	return result * alpha / _2pi<T>;
}

template ARMA_REAL_TYPE
arma::apmath::owen_t(ARMA_REAL_TYPE x, ARMA_REAL_TYPE alpha);
