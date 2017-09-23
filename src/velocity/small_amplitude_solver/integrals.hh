#ifndef PRESSURE_HH
#define PRESSURE_HH

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "apmath/delaunay_interpolation.hh"
#include "physical_constants.hh"
#include "types.hh"
#include "wave.hh"

namespace arma {

	namespace bits {

		template<class T>
		struct Integral_1 {

			typedef const arma::Array3D<T>& V;

			inline
			Integral_1(
				V wavenum_x_,
				V wavenum_,
				V zeta_,
				V alpha_,
				int i_,
				int k_
			):
			wavenum_x(wavenum_x_),
			wavenum(wavenum_),
			zeta(zeta_),
			alpha(alpha_),
			idx_t(i_),
			idx_y(k_)
			{}

			T operator()(int j1) const {
				T sum = 0;
				int j0 = 0;
				for (int j=j0; j<=j1; ++j) {
					const T a = alpha(idx_t, j, idx_y);
					sum += (wavenum_x(idx_t, j, idx_y)*zeta(idx_t, j, idx_y)
							+ wavenum(idx_t, j, idx_y)*a)
							/ (T(1) + a*a);
				}
				return sum;
			}

			V wavenum_x, wavenum, zeta, alpha;
			int idx_t, idx_y;
		};

		template<class T>
		struct Integral_2 {

			typedef const arma::Array3D<T>& V;

			inline
			Integral_2(
				V wavenum_t_,
				V wavenum_,
				V zeta_,
				V zeta_t_,
				V alpha_,
				V alpha_t_,
				int i_,
				int k_,
				Integral_1<T>& int1
			):
			wavenum_t(wavenum_t_), wavenum(wavenum_), zeta(zeta_), zeta_t(zeta_t_),
			alpha(alpha_), alpha_t(alpha_t_),
			idx_t(i_), idx_y(k_), integral_1(int1)
			{}

			T operator()(int j1) const {
				T sum = 0;
				int j0 = 0;
				for (int j=j0; j<=j1; ++j) {
					T a = alpha(idx_t,j,idx_y);
					sum += (wavenum_t(idx_t,j,idx_y)*zeta(idx_t,j,idx_y)
								+ wavenum(idx_t,j,idx_y)*zeta_t(idx_t,j,idx_y)
								+ a*alpha_t(idx_t,j,idx_y))
							/ sqrt(T(1) + a*a)
							* exp(integral_1(j));
				}
				return sum;
			}

			V wavenum_t, wavenum, zeta, zeta_t, alpha, alpha_t;
			Integral_1<T>& integral_1;
			int idx_t, idx_y;
		};


		template<class T>
		struct Event {
			Event(): x(0), y(0), period(0), elevation(0), slope(0) {}
			Event(size_t i, size_t j, T p, T z, T a):
			x(i), y(j), period(p), elevation(z), slope(a)
			{}

			T frequency() const noexcept { return T(1) / period; }

			size_t x, y;
			T period, elevation, slope;

		};

		template<class T>
		std::vector<Wave<T>>
		factor_waves_x(
			const Discrete_function<T,3>& z,
			T min_k,
			T max_k,
			int t
		) {
			using std::abs;
			using constants::_2pi;
			const Shape3D& zsize = z.shape();
			const Vec3D<T> zdelta = z.grid().delta();
			const T dt = zdelta(0);
			std::vector<Wave<T>> waves;
			std::vector<Event<T>> events;
			const int nt = zsize(0);
			const int nx = zsize(1);
			const int ny = zsize(2);
		//	for (int t=0; t<nt; t++) {
			for (int y=0; y<ny; y++) {
				for (int x=1; x<nx-1; x++) {
					const T z0 = z(t, x, y);
					const T z1 = z(t, x-1, y);
					const T z2 = z(t, x+1, y);
					const T dw1 = z0 - z1;
					const T dw2 = z0 - z2;
					if (!((dw1 > T(0))^(dw2 > T(0)))) {
						const T a = T(-0.5)*(dw1 + dw2)/(dt*dt);
						const T b = dw1/dt - a*dt*(T(2)*t - T(1));
						const T c = z0 - t*dt*(a*t*dt + b);
						const T Tex = T(-0.5)*b/a;
						const T Wex = c + Tex*(b + a*Tex);
						const T alpha = (z2 - z1) / (dt + dt);
						events.emplace_back(x, y, Tex, Wex, alpha);
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
						T g = 9.8;
						int x = event1.x;
						int y = event1.y;
						if (event2.elevation > event1.elevation) {
							x = event2.x;
							y = event2.y;
						}
						T k = w*w/g;
						T height = abs(event1.elevation - event2.elevation);
						T length = abs(event1.period - event2.period)*T(2);
		//				T s = height / length;
						if (k > min_k && k < max_k) {
							waves.emplace_back(x, y, k, height, length);
						}
					}
					event2 = event1;
					event1 = events[i];
					j++;
				}
			}
			events.clear();
			return waves;
		}

		template<class T>
		std::vector<Wave<T>> factor_waves_y(
			const Discrete_function<T,3>& z,
			T min_k,
			T max_k,
			int t
		) {
			using std::abs;
			using constants::_2pi;
			const Shape3D& zsize = z.shape();
			const Vec3D<T> zdelta = z.grid().delta();
			const T dt = zdelta(0);
			std::vector<Wave<T>> waves;
			std::vector<Event<T>> events;
			const int nt = zsize(0);
			const int nx = zsize(1);
			const int ny = zsize(2);
			for (int x=0; x<nx; x++) {
				for (int y=1; y<ny-1; y++) {
					const T z0 = z(t, x, y);
					const T z1 = z(t, x, y-1);
					const T z2 = z(t, x, y+1);
					const T dw1 = z0 - z1;
					const T dw2 = z0 - z2;
					if (!((dw1 > T(0))^(dw2 > T(0)))) {
						const T a = T(-0.5)*(dw1 + dw2)/(dt*dt);
						const T b = dw1/dt - a*dt*(T(2)*t - T(1));
						const T c = z0 - t*dt*(a*t*dt + b);
						const T Tex = T(-0.5)*b/a;
						const T Wex = c + Tex*(b + a*Tex);
						const T alpha = (z2 - z1) / (dt + dt);
						events.emplace_back(x, y, Tex, Wex, alpha);
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
						T g = 9.8;
						size_t x = event1.x;
						size_t y = event1.y;
						T alpha = event1.slope;
						T zeta = event1.elevation;
						if (event2.elevation > event1.elevation) {
							x = event2.x;
							y = event2.y;
							alpha = event2.slope;
							zeta = event2.elevation;
						}
						T k = w*w/g;
						T height = abs(event1.elevation - event2.elevation);
						T length = abs(event1.period - event2.period)*T(2);
		//				T s = height / length;
						if (k > min_k && k < max_k) {
							waves.emplace_back(x, y, k, height, length);
						}
					}
					event2 = event1;
					event1 = events[i];
					j++;
				}
			}
			events.clear();
			return waves;
		}

		template<class T>
		Array2D<T>
		wave_number(
			const Discrete_function<T,3>& z,
			T min_k,
			T max_k,
			int idx_t
		) {
			Vec3D<T> delta = z.grid().delta();
			const int nx = z.extent(1);
			const int ny = z.extent(2);
			Shape2D ksize(blitz::shape(nx, ny));
			Array2D<T> wavenum(ksize);
			std::ofstream out("waves");
			std::vector<Wave<T>> waves_x = factor_waves_x(z, min_k, max_k, idx_t);
			std::vector<Wave<T>> waves_y = factor_waves_y(z, min_k, max_k, idx_t);
			std::vector<Wave<T>> waves;
			std::copy(waves_x.begin(), waves_x.end(), std::back_inserter(waves));
			std::copy(waves_y.begin(), waves_y.end(), std::back_inserter(waves));
			out << waves;
			if (waves.size() > 2) {
				Delaunay_interpolation<T> interpolation;
				for (const Wave<T>& w : waves) {
					interpolation.insert(w.x(), w.y(), w.wave_number());
				}
				for (int j=0; j<nx; ++j) {
					for (int k=0; k<ny; ++k) {
						wavenum(j,k) = interpolation(j, k);
					}
				}
			}
			return wavenum;
		}


		template<class T>
		Array2D<T>
		velocity_field(
			const Discrete_function<T,3>& surface,
			int idx_t,
			int idx_y
		) {
			using constants::_2pi;
			const Vec3D<T>& delta = surface.grid().delta();
			const Shape3D& size = surface.shape();
			const int k_count = 40;
			const Domain<T,1> krange(
				Vec1D<T>(_2pi<T>/T(20)),
				Vec1D<T>(_2pi<T>/T(2)),
				k_count
			);
			Shape3D ksize = size;
			Array2D<T> wavenum = wave_number(
				surface,
				krange.lbound(),
				krange.ubound()
			);
			Array2D<T> wavenum_t = derivative<0>(wavenum, delta, idx_t);
			Array2D<T> wavenum_x = derivative<1>(wavenum, delta, idx_t);
			Array2D<T> zeta_t = derivative<0>(surface, delta, idx_t);
			Array2D<T> alpha = derivative<1>(surface, delta, idx_t);
			Array2D<T> alpha_t = derivative<0>(alpha, delta, idx_t);
			Array2D<T> alpha_x = derivative<1>(alpha, delta, idx_t);
			std::clog << "lambda count = " << k_count << std::endl;

			{ std::ofstream out("alpha"); out << alpha; }
			{ std::ofstream out("alpha_x"); out << alpha_x; }
			{ std::ofstream out("alpha_t"); out << alpha_t; }

			// compute reference velocity field using assumptions
			// from small amplitude wave theory
			Integral_1<T> integral_1(
				wavenum_x,
				wavenum,
				surface,
				alpha,
				idx_t,
				idx_y
			);
			Integral_2<T> integral_2(
				wavenum_t,
				wavenum,
				surface,
				zeta_t,
				alpha,
				alpha_t,
				idx_t,
				idx_y,
				integral_1
			);

			const int nx = size(1);
			const int ny = size(2);
			Array2D<T> u(blitz::shape(nx, ny));
			for (int i=0; i<nx; ++i) {
				for (int j=0; j<ny; ++j) {
					using std::sqrt;
					using std::exp;
					u(i,j) = - (T(1)/sqrt(T(1) + alpha(i, j)*alpha(i, j)))
							* exp(-integral_1(i))
							* integral_2(i);
				}
			}

			{ std::ofstream out("u0"); out << u; }

			return u;
		}

	}
}

#endif // PRESSURE_HH
