#include "acf_generator.hh"

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <fstream>

#include "arma.hh"
#include "chop.hh"
#include "params.hh"
#include "plain_wave.hh"
#include "stats/statistics.hh"

namespace {

	template <class T, class Function>
	inline arma::Array3D<T>
	generate_wavy_surface(
		Function func,
		arma::Domain<T,3> grid,
		T amplitude,
		T kx,
		T ky,
		T velocity,
		T phase
	) {
		arma::Array3D<T> result(grid.shape());
		const int ni = grid.num_points(0);
		const int nj = grid.num_points(1);
		const int nk = grid.num_points(2);
		#if ARMA_OPENMP
		#pragma omp parallel for collapse(3)
		#endif
		for (int i=0; i<ni; ++i) {
			for (int j=0; j<nj; ++j) {
				for (int k=0; k<nk; ++k) {
					const T t = grid(i, 0);
					const T x = grid(j, 1);
					const T y = grid(k, 2);
					result(i,j,k) =
						func(amplitude, kx, ky, velocity, phase, x, y, t);
				}
			}
		}
		return result;
	}

	template <class T>
	inline arma::Array3D<T>
	generate_wavy_surface(
		arma::generator::Plain_wave_profile profile,
		arma::Domain<T,3> grid,
		T amplitude,
		T kx,
		T ky,
		T velocity,
		T phase
	) {
		using namespace arma::generator;
		switch (profile) {
		case Plain_wave_profile::Sine:
			return generate_wavy_surface(
				sine_wave<T>,
				grid,
				amplitude,
				kx,
				ky,
				velocity,
				phase
			);
		case Plain_wave_profile::Cosine:
			return generate_wavy_surface(
				cosine_wave<T>,
				grid,
				amplitude,
				kx,
				ky,
				velocity,
				phase
			);
		case Plain_wave_profile::Stokes:
			return generate_wavy_surface(
				stokes_wave<T>,
				grid,
				amplitude,
				kx,
				ky,
				velocity,
				phase
			);
		case Plain_wave_profile::Standing_wave:
			return generate_wavy_surface(
				standing_wave<T>,
				grid,
				amplitude,
				kx,
				ky,
				velocity,
				phase
			);
		default:
			throw std::invalid_argument("bad wave profile function");
		}
	}

	template <class T, class Function>
	inline arma::Array3D<T>
	generate_field(arma::Domain<T, 3> grid, Function func) {
		arma::Array3D<T> result(grid.shape());
		const int ni = grid.num_points(0);
		const int nj = grid.num_points(1);
		const int nk = grid.num_points(2);
		#if ARMA_OPENMP
		#pragma omp parallel for collapse(3)
		#endif
		for (int i=0; i<ni; ++i) {
			for (int j=0; j<nj; ++j) {
				for (int k=0; k<nk; ++k) {
					const T t = grid(i, 0);
					const T x = grid(j, 1);
					const T y = grid(k, 2);
					result(i,j,k) = func(x, y, t);
				}
			}
		}
		return result;
	}

}


template <class T>
typename arma::generator::ACF_generator<T>::array_type
arma::generator::ACF_generator<T>
::generate() {
	using arma::stats::variance;
	using blitz::RectDomain;
	Array3D<T> wave;
	domain_type domain;
	auto p = this->generate_optimal_wavy_surface();
	wave.reference(p.first);
	domain = p.second;
	wave.reference(this->add_exponential_decay(wave, domain));
	std::clog << "variance(wave)=" << variance(wave) << std::endl;
	Array3D<T> acf = arma::auto_covariance(wave);
	acf.resizeAndPreserve(acf.shape()/2);
	std::clog << "acf(0,0,0)=" << acf(0,0,0) << std::endl;
//	acf.resizeAndPreserve(chop_right(acf, acf(0,0,0)*this->_chopepsilon));
//	std::clog << "acf.shape()=" << acf.shape() << std::endl;
	std::ofstream("zeta") << acf;
	const T r = this->_nwaves;
	array_type result;
	result.reference(acf);
	result.setgrid(Grid<T,3>{acf.shape(), {r,r,r}});
	return result;
}

template <class T>
typename arma::generator::ACF_generator<T>::array_and_domain
arma::generator::ACF_generator<T>
::generate_optimal_wavy_surface() {
	using arma::stats::variance;
	using blitz::all;
	const T r = this->_nwaves;
	T var0 = -1;
	T var = -1;
	const T eps = this->_varepsilon;
	Shape3D wave_shape(2,2,2);
	Array3D<T> surface;
	Domain<T,3> grid;
	while (all(wave_shape < 128)) {
		grid = Domain<T,3>{{-r,-r,-r}, {r,r,r}, wave_shape+1};
		surface.reference(
			generate_wavy_surface(
				this->_func,
				grid,
				this->_amplitude,
				this->_wavenum(0),
				this->_wavenum(1),
				this->_velocity,
				T(0)
			)
		);
		var0 = var;
		var = variance(surface);
//		#ifndef NDEBUG
		std::clog
		    << __func__
		    << ": var=" << var
		    << ",shape=" << wave_shape
		    << ",eps=" << std::abs(var-var0)
		    << std::endl;
//		#endif
		if (!(var0 < T(0)) && std::abs(var-var0) < eps) {
			break;
		}
		wave_shape *= 2;
	}
	std::clog
	    << "wave shape = " << grid.shape()
	    << std::endl;
	return std::make_pair(surface, grid);
}

template <class T>
arma::Array3D<T>
arma::generator::ACF_generator<T>
::add_exponential_decay(Array3D<T> wave, const domain_type& domain) {
	using arma::stats::variance;
	using std::sqrt;
	Array3D<T> wave_exp{
		wave *
		generate_field(
			domain,
			[this] (T x, T y, T t) {
			    using blitz::abs;
			    using blitz::sum;
			    using std::abs;
			    using std::exp;
				Vec3D<T> txy(t,x,y);
			    return exp(-(sum(abs(txy * this->_alpha))));
			}
		)
	};
	wave_exp *= sqrt(variance(wave) / variance(wave_exp));
	return wave_exp;
}

template <class T>
std::istream&
arma::generator::operator>>(std::istream& in, ACF_generator<T>& rhs) {
	sys::parameter_map params {
		{
			{"func", sys::make_param(rhs._func)},
			{"amplitude", sys::make_param(rhs._amplitude)},
			{"velocity", sys::make_param(rhs._velocity)},
			{"alpha", sys::make_param(rhs._alpha)},
			{"beta", sys::make_param(rhs._wavenum)},
			{"nwaves", sys::make_param(rhs._nwaves)},
		},
		true
	};
	in >> params;
	return in;
}

template class arma::generator::ACF_generator<ARMA_REAL_TYPE>;

template std::istream&
arma::generator::operator>>(
	std::istream& in,
	ACF_generator<ARMA_REAL_TYPE>& rhs
);
