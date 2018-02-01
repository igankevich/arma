#ifndef STATS_WAVES_HH
#define STATS_WAVES_HH

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

		enum struct Wave_feature_type {
			Crest,
			Trough
		};

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
		struct Wave_field {

			typedef std::vector<Wave<T>> wave_vector;

			explicit
			Wave_field(Array3D<T> elevation, const Grid<T,3>& grid);

			Array1D<T>
			periods() const;

			Array1D<T>
			lengths() const;

			Array1D<T>
			lengths_x() const;

			Array1D<T>
			lengths_y() const;

			Array1D<T>
			heights() const;

			Array1D<T>
			heights_x() const;

			Array1D<T>
			heights_y() const;

		private:

			void
			extract_waves_t(Array3D<T> elevation, const Grid<T,3>& grid);

			void
			extract_waves_x(Array3D<T> elevation, const Grid<T,3>& grid);

			void
			extract_waves_y(Array3D<T> elevation, const Grid<T,3>& grid);

			void
			extract_waves_x2(Array3D<T> elevation);

			void
			extract_waves_y2(Array3D<T> elevation);

			wave_vector _wavest;
			wave_vector _wavesx;
			wave_vector _wavesy;
		};

		template <class T>
		std::vector<Wave_feature<T>>
		find_extrema(Array1D<T> elevation, const Grid<T,1>& grid);

		template <class T>
		std::vector<Wave<T>>
		factor_waves(const std::vector<Wave_feature<T>>& features);

	}

}

#endif // STATS_WAVES_HH
