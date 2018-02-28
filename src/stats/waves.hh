#ifndef STATS_WAVES_HH
#define STATS_WAVES_HH

#include "domain.hh"
#include "grid.hh"
#include "types.hh"

#include <algorithm>
#include <cmath>
#include <iosfwd>
#include <iterator>
#include <tuple>
#include <vector>

namespace arma {

	namespace stats {

		/// Wavy surface extrema type.
		enum struct Wave_feature_type {
			Crest,
			Trough
		};

		/// Wavy surface extrema locations.
		template <class T>
		struct Wave_feature {
			Wave_feature(T xx, T zz, Wave_feature_type t):
			x(xx), z(zz), type(t) {}
			T x;
			T z;
			Wave_feature_type type;
		};

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const Wave_feature<T>& rhs);

		/// \brief A wave with height and period.
		template <class T>
		struct Wave {

			static constexpr const int
				cheight = 0,
				clength = 1,
				cperiod = 1;

			Wave() = default;

			inline
			Wave(T height, T period):
			_height(height),
			_period(period)
			{}

			inline T
			height() const noexcept {
				return _height;
			}

			inline T
			period() const noexcept {
				return _period;
			}

		private:
			T _height = 0;
			T _period = 0;
		};

		/// \brief Extracts individual \link Wave waves \endlink
		/// along each dimension.
		template <class T>
		class Wave_field {

		private:
			Array1D<Wave<T>> _wavest;
			Array1D<Wave<T>> _wavesx;
			Array1D<Wave<T>> _wavesy;

		public:

			/// \param kradius The size of Gaussian kernel.
			explicit
			Wave_field(
				Array3D<T> elevation,
				const Grid<T, 3>& grid,
				int kradius
			);

			inline Array1D<T>
			periods() const {
				return this->_wavest[Wave<T>::cperiod].copy();
			}

			Array1D<T>
			lengths() const;

			Array1D<T>
			heights() const;

			inline Array1D<T>
			lengths_x() const {
				return this->_wavesx[Wave<T>::clength].copy();
			}

			inline Array1D<T>
			lengths_y() const {
				return this->_wavesy[Wave<T>::clength].copy();
			}

			inline Array1D<T>
			heights_x() const {
				return this->_wavesx[Wave<T>::cheight].copy();
			}

			inline Array1D<T>
			heights_y() const {
				return this->_wavesy[Wave<T>::cheight].copy();
			}

		};

		/**
		\brief Find locations of wave crests and troughs.
		\date 2018-02-02
		\author Ivan Gankevich
		*/
		template <class T>
		std::vector<Wave_feature<T>>
		find_extrema(Array1D<T> elevation, const Domain<T,1>& grid);

		/**
		\brief Find locations of wave crests and troughs.
		\date 2018-02-02
		\author Ivan Gankevich
		*/
		template <class T>
		std::vector<Wave_feature<T>>
		find_extrema(Array1D<T> elevation, const Grid<T,1>& grid) {
			return find_extrema(elevation, Domain<T,1>(grid));
		}

		/**
		\brief
		Transform wave features (extremas) to individual waves.
		\date 2018-02-02
		\author Ivan Gankevich
		*/
		template <class T>
		std::vector<Wave<T>>
		find_waves(const std::vector<Wave_feature<T>>& features);

		/**
		\brief
		Extract individual waves by smoothing the wavy surface
		and finding all extrema.
		\param r Gaussian kernel radius
		\date 2018-02-02
		\author Ivan Gankevich
		*/
		template <class T>
		std::vector<Wave<T>>
		find_waves(Array1D<T> elevation, Domain<T,1> grid, int r=11);

		/**
		\brief
		Generate Gaussian kernel with radius \f$r\f$ and
		standard deviation \f$\sigma\f$.
		\param r kernel radius
		\param sigma standard deviation
		\date 2018-02-02
		\author Ivan Gankevich

		- The size of the kernel equals \f$2r+1\f$.
		- The kernel formula is \f$\exp\left[-\frac{x^2}{2\sigma}\right]\f$.
		- The sum of all points equals 1.
		*/
		template <class T>
		Array1D<T>
		gaussian_kernel(int r, T sigma);

		/**
		\brief
		The same as \link gaussian_kernel\endlink but with \f$\sigma=r/2\f$.
		\date 2018-02-02
		\author Ivan Gankevich
		*/
		template <class T>
		arma::Array1D<T>
		gaussian_kernel(int r) {
			return gaussian_kernel(r, T(0.5)*r);
		}

		/**
		\brief
		Convolve the signal with the kernel taking real part of the result.
		\date 2018-02-02
		\author Ivan Gankevich
		*/
		template <class T, int N>
		blitz::Array<T,N>
		filter(blitz::Array<T,N> data, blitz::Array<T,N> kernel);

		/**
		\brief
		Smooth elevation field by convolving with Gaussian kernel with
		radius \f$r\f$.
		\param elevation the array of wavy surface elevation
		\param grid defines delta
		\param r the radius of Gaussian kernel
		\date 2018-02-02
		\author Ivan Gankevich
		\see gaussian_kernel
		*/
		template <class T>
		void
		smooth_elevation(
			Array1D<T>& elevation,
			Domain<T, 1>& grid,
			int r
		);

		template <class T>
		Array3D<T>
		frequency_amplitude_spectrum(Array3D<T> rhs, const Grid<T,3>& grid);

	}

}

BZ_DECLARE_MULTICOMPONENT_TYPE(
	::arma::stats::Wave<ARMA_REAL_TYPE>,
	ARMA_REAL_TYPE,
	2
);

#endif // STATS_WAVES_HH
