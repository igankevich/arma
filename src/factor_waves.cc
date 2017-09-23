#include "factor_waves.hh"

#include <limits>

#include "types.hh"
#include "physical_constants.hh"

namespace {

	template<class T>
	struct Event {
		Event(): x(0), y(0), period(0), elevation(0) {}
		Event(int i, int j, T p, T z):
		x(i), y(j), period(p), elevation(z)
		{}
		T frequency() const noexcept { return T(1) / period; }
		int x, y;
		T period, elevation;
	};

	template<class T>
	std::pair<T,T>
	factor_waves_dim(blitz::Array<T, 2> z, int t, T dt) {
		using namespace arma;
		using std::abs;
		using constants::_2pi;
		using constants::g;
		const Shape2D& zsize = z.shape();
		T min_k = std::numeric_limits<T>::max();
		T max_k = std::numeric_limits<T>::min();
		std::vector<Event<T>> events;
		const int nx = zsize(0);
		const int ny = zsize(1);
		for (int y=0; y<ny; y++) {
			for (int x=1; x<nx-1; x++) {
				const T z0 = z(x, y);
				const T z1 = z(x-1, y);
				const T z2 = z(x+1, y);
				const T dw1 = z0 - z1;
				const T dw2 = z0 - z2;
				if (!((dw1 > T(0))^(dw2 > T(0)))) {
					const T a = T(-0.5)*(dw1 + dw2)/(dt*dt);
					const T b = dw1/dt - a*dt*(T(2)*t - T(1));
					const T c = z0 - t*dt*(a*t*dt + b);
					const T Tex = T(-0.5)*b/a;
					const T Wex = c + Tex*(b + a*Tex);
					events.emplace_back(x, y, Tex, Wex);
				}
			}
		}
		size_t N = events.size();
		Event<T> event1 = events[0];
		Event<T> event2;
		size_t j = 0;
		for (size_t i=1; i<N; i++) {
			if (!((event1.elevation > T(0))^(events[i].elevation > T(0)))) {
				if (abs(event1.elevation) < abs(events[i].elevation)) {
					event1 = events[i];
				}
			} else {
				if (j > 0) {
					T w = _2pi<T> / abs((event1.period - event2.period)*T(2));
					T k = w*w/g<T>;
					if (k < min_k) {
						min_k = k;
					}
					if (k > max_k) {
						max_k = k;
					}
				}
				event2 = event1;
				event1 = events[i];
				j++;
			}
		}
		return std::make_pair(min_k, max_k);
	}

}

template<class T>
arma::Domain<T,2>
arma::factor_waves(Array2D<T> z, int t, T dt) {
	std::pair<T,T> wn_range_x = factor_waves_dim(z, t, dt);
	std::pair<T,T> wn_range_y = factor_waves_dim(z.transpose(1, 0), t, dt);
	return Domain<T,2>{
		{wn_range_x.first,wn_range_y.first},
		{wn_range_x.second,wn_range_y.second},
		{2, 2}
	};
}


template arma::Domain<ARMA_REAL_TYPE,2>
arma::factor_waves(
	arma::Array2D<ARMA_REAL_TYPE> z,
	int t,
	ARMA_REAL_TYPE dt
);
