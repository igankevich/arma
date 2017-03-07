#ifndef LINEAR_VELOCITY_FIELD_HH
#define LINEAR_VELOCITY_FIELD_HH

#include <cmath>
#include <complex>
#include <stdexcept>
#include <cassert>
#include "velocity_potential_field.hh"
#include "fourier.hh"
#include "physical_constants.hh"
#include "derivative.hh"

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

	protected:
		void
		precompute(const Array3D<T>& zeta) override {
			_fft.init(size2(zeta.extent(1), zeta.extent(2)));
		}

		Array2D<T>
		compute_velocity_field_2d(
			const Array3D<T>& zeta,
			const size2 arr_size,
			const T z,
			const int idx_t
		) override {
			using blitz::all;
			using blitz::isfinite;
			typedef std::complex<T> Cmplx;
			/**
			1. Compute multiplier.
			\f[
			\text{mult}(u, v) =
				-4\pi \frac{ \cosh\left(|\vec{k}|(z + h)\right) }
				        { |\vec{k}|\cosh\left(|\vec{k}|h\right) }
			\f]
			*/
			const Domain<T,2> wngrid(this->_wnmax, arr_size);
			Array2D<T> mult = low_amp_window_function(wngrid, z);
			if (!all(isfinite(mult))) {
				std::clog << "Infinite/NaN multiplier. Try to increase minimal z "
					"coordinate at which velocity potential is calculated, or "
					"decrease water depth. Here z="
					<< z << ",depth=" << this->_depth << '.' << std::endl;
				throw std::runtime_error("bad multiplier");
			}
			/// 2. Compute \f$\zeta_t\f$.
			Array2D<Cmplx> phi = derivative<0,T,Cmplx>(zeta, idx_t);
			/**
			3. Compute Fourier transforms.
			\f[
			\phi(x,y,z,t) =
				\text{Re}\left\{
					\mathcal{F}_{x,y}^{-1}\left\{
						\text{mult}(u, v) \mathcal{F}_{u,v}\left\{\zeta_t\right\}
					\right\}
				\right\}
			\f]
			*/
			return blitz::real(_fft.backward(_fft.forward(phi) *= mult));
		}

	private:
		Array2D<T>
		low_amp_window_function(const Domain<T,2>& wngrid, const T z) {
			using std::cosh;
			using blitz::length;
			using constants::_2pi;
			const T h = this->_depth;
			Array2D<T> result(wngrid.num_points());
			const int nx = wngrid.num_points(0);
			const int ny = wngrid.num_points(1);
			for (int i=0; i<nx; ++i) {
				for (int j=0; j<ny; ++j) {
					const T l = _2pi<T> * length(wngrid({i,j}));
					const T numerator = cosh(l*(z + h));
					const T denominator = l*cosh(l*h);
					result(i, j) = _2pi<T>
						* T(-2)
						* bits::div_or_nought(numerator, denominator);
				}
			}
			return result;
		}


	};

}

#endif // LINEAR_VELOCITY_FIELD_HH
