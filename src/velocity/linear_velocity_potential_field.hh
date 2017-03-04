#ifndef LINEAR_VELOCITY_FIELD_HH
#define LINEAR_VELOCITY_FIELD_HH

#include <cmath>
#include <complex>
#include <stdexcept>
#include "velocity_potential_field.hh"
#include "fourier.hh"

namespace arma {

	namespace bits {

		template <class T>
		inline T
		div_or_nought(T lhs, T rhs) noexcept {
			T result = lhs / rhs;
			if (!std::isfinite(result)) {
				result = T(0);
			}
			return result;
		}

	}

	/// Linear wave theory formula to compute velocity potential field.
	template<class T>
	class Linear_velocity_potential_field: public Velocity_potential_field<T> {

		Fourier_transform<std::complex<T>, 2> _fft;
		static constexpr const T _2pi = T(2) * M_PI;

	protected:
		Array2D<T>
		compute_velocity_field_2d(
			Array3D<T>& zeta,
			const size2 arr_size,
			const T z,
			const int idx_t
		) override {
			using blitz::all;
			using blitz::isfinite;
			/**
			1. Compute multiplier.
			\f[
			\text{mult}(u, v) =
				-2 \frac{ \cosh\left(|\vec{k}|(z + h)\right) }
				        { |\vec{k}|\cosh\left(|\vec{k}|h\right) }
				=
				-2 \frac{ e^{|\vec{k}|z} + e^{-|\vec{k}|(z + 2h)} }
				        { |\vec{k}| \left(1 + e^{-2|\vec{k}|h}\right) }
			\f]
			*/
			const Domain<T,2> wngrid(this->_wnmax, arr_size);
			const T h = this->_depth;
			Array2D<T> mult(wngrid.num_points());
			const int nx = wngrid.num_points(0);
			const int ny = wngrid.num_points(1);
			for (int i=0; i<nx; ++i) {
				for (int j=0; j<ny; ++j) {
					const T l = _2pi * blitz::length(wngrid({i,j}));
//					const T expm2lh = std::exp(T(-2)*l*h);
//					const T expmlz = std::exp(-l*z);
//					const T explz = std::exp(l*z);
//					const T numerator = explz + expmlz*expm2lh;
//					const T denominator = l*(T(1) + expm2lh);
					const T numerator = std::cosh(l*(z + h));
					const T denominator = l*std::cosh(l*h);
					mult(i, j) = _2pi *T(-2) * bits::div_or_nought(numerator, denominator);
				}
			}
			//blitz::rotate(mult, mult.extent()/2);
			if (!all(isfinite(mult))) {
				std::clog << __FILE__ << ':' << __LINE__ << ':' << __func__
					<< ": infinite/NaN multiplier. Try to increase minimal z "
					"coordinate at which velocity potential is calculated, or "
					"decrease water depth. Here z="
					<< z << ",depth=" << h << '.' << std::endl;
				throw std::runtime_error("bad multiplier");
			}
			/// 2. Compute \f$\zeta_t\f$.
			Array2D<std::complex<T>> phi(arr_size);
			// TODO Implement in a separate function with proper handling of borders.
			for (int i=0; i<nx; ++i) {
				for (int j=0; j<ny; ++j) {
					phi(i,j) = T(0.5)*(zeta(idx_t+1,i,j) - zeta(idx_t-1,i,j));
				}
			}
			/**
			3. Compute Fourier transforms.
			\f[
			\phi(x,y,z,t) =
				\mathcal{F}_{x,y}^{-1}\left\{
					\text{mult}(u, v) \mathcal{F}_{u,v}\left\{\zeta_t\right\}
				\right\}
			\f]
			*/
			#if ARMA_OPENMP
			#pragma omp critical
			#endif
			_fft.init(arr_size);
			return blitz::real(_fft.backward(_fft.forward(phi) *= mult));
		}

	};

}

#endif // LINEAR_VELOCITY_FIELD_HH
