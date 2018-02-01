#include "waves.hh"

#include <ostream>

namespace {

	template <class T, class Result>
	Result
	getperiods(
		const typename arma::stats::Wave_field<T>::wave_vector& rhs,
		Result result
	) {
		return std::transform(
			rhs.begin(), rhs.end(), result,
			[](const arma::stats::Wave<T>& wave) { return wave.period(); });
	}

	template <class T, class Result>
	static Result
	getheights(
		const typename arma::stats::Wave_field<T>::wave_vector& rhs,
		Result result
	) {
		return std::transform(
			rhs.begin(), rhs.end(), result,
			[](const arma::stats::Wave<T>& wave) { return wave.height(); });
	}

	template <class T, class Result>
	void
	copy_waves_t(const T* elevation, size_t n, Result result) {
		enum Type { Crest, Trough };
		std::vector<std::tuple<T, T, Type>> peaks;
		for (size_t i = 1; i < n - 1; ++i) {
			const T x1 = T(i - 1) / (n - 1);
			const T x2 = T(i) / (n - 1);
			const T x3 = T(i + 1) / (n - 1);
			const T z1 = elevation[i - 1];
			const T z2 = elevation[i];
			const T z3 = elevation[i + 1];
			const T dz1 = z2 - z1;
			const T dz2 = z2 - z3;
			const T dz3 = z1 - z3;
			if ((dz1 > 0 && dz2 > 0) || (dz1 < 0 && dz2 < 0)) {
				const T a = T(-0.5) * (x3 * dz1 + x2 * dz3 - x1 * dz2);
				const T b =
					T(-0.5) * (-x3 * x3 * dz1 + x1 * x1 * dz2 - x2 * x2 * dz3);
				const T c = T(-0.5) *
							(x1 * x3 * 2 * z2 + x2 * x2 * (x3 * z1 - x1 * z3) +
							 x2 * (-x3 * x3 * z1 + x1 * x1 * z3));
				peaks.emplace_back(-b / (T(2) * a),
								   -(b * b - T(4) * a * c) / (T(4) * a),
								   dz1 < 0 ? Crest : Trough);
			}
		}
		int trough_first = -1;
		int crest = -1;
		int trough_last = -1;
		int npeaks = peaks.size();
		for (int i = 0; i < npeaks; ++i) {
			const auto& peak = peaks[i];
			if (std::get<2>(peak) == Trough) {
				if (trough_first == -1) {
					trough_first = i;
				} else if (crest != -1) {
					trough_last = i;
				}
			} else {
				if (trough_first != -1) { crest = i; }
			}
			if (trough_first != -1 && crest != -1 && trough_last != -1) {
				const T elev_trough_first = std::get<1>(peaks[trough_first]);
				const T elev_crest = std::get<1>(peaks[crest]);
				const T elev_trough_last = std::get<1>(peaks[trough_last]);
				const T height =
					std::max(std::abs(elev_crest - elev_trough_first),
							 std::abs(elev_crest - elev_trough_last));
				const T time_first = std::get<0>(peaks[trough_first]);
				const T time_last = std::get<0>(peaks[trough_last]);
				const T period = time_last - time_first;
				*result = arma::stats::Wave<T>(height, period * (n - 1));
				++result;
				trough_first = trough_last;
				crest = -1;
				trough_last = -1;
			}
		}
	}

	template <class T, class Result>
	void
	copy_waves_x(const T* elevation, size_t n, Result result) {
		const T dt = 1;
		std::vector<T> Tex, Wex;
		for (size_t i = 1; i < n - 1; ++i) {
			const T e0 = elevation[i];
			const T dw1 = e0 - elevation[i - 1];
			const T dw2 = e0 - elevation[i + 1];
			if ((dw1 > 0 && dw2 > 0) || (dw1 < 0 && dw2 < 0)) {
				T a = -T(0.5) * (dw1 + dw2) / (dt * dt);
				T b = dw1 / dt - a * dt * (2 * i - 1);
				T c = e0 - i * dt * (a * i * dt + b);
				T tex = -T(0.5) * b / a;
				T wex = c + tex * (b + a * tex);
				Tex.push_back(tex);
				Wex.push_back(wex);
			}
		}
		if (!Tex.empty()) {
			const int N = std::min(Tex.size() - 1, size_t(100));
			T Wexp1 = Wex[0];
			T Texp1 = Tex[0];
			T Wexp2 = 0, Texp2 = 0;
			int j = 0;
			for (int i = 1; i < N; ++i) {
				if (!((Wexp1 > T(0)) ^ (Wex[i] > T(0)))) {
					if (std::abs(Wexp1) < std::abs(Wex[i])) {
						Wexp1 = Wex[i];
						Texp1 = Tex[i];
					}
				} else {
					if (j >= 1) {
						T period = (Texp1 - Texp2) * T(2);
						T height = std::abs(Wexp1 - Wexp2);
						*result = arma::stats::Wave<T>(height, period);
						++result;
					}
					Wexp2 = Wexp1;
					Texp2 = Texp1;
					Wexp1 = Wex[i];
					Texp1 = Tex[i];
					j++;
				}
			}
		}
	}

	template <class C1, class C2>
	void
	push_back_all(C1& lhs, const C2& rhs) {
		lhs.insert(lhs.end(), rhs.begin(), rhs.end());
	}

}


template <class T>
arma::stats::Wave_field<T>::Wave_field(
	Array3D<T> elevation,
	const Grid<T, 3>& grid
) {
	extract_waves_t(elevation, grid);
	extract_waves_x(elevation, grid);
	extract_waves_y(elevation, grid);
}

template <class T>
arma::Array1D<T>
arma::stats::Wave_field<T>::periods() const {
	Array1D<T> result(_wavest.size());
	getperiods<T>(_wavest, result.begin());
	return result;
}

template <class T>
arma::Array1D<T>
arma::stats::Wave_field<T>::lengths() const {
	Array1D<T> result(this->_wavesx.size() + this->_wavesy.size());
	getperiods<T>(this->_wavesy, getperiods<T>(this->_wavesx, result.begin()));
	return result;
}

template <class T>
arma::Array1D<T>
arma::stats::Wave_field<T>::lengths_x() const {
	Array1D<T> result(this->_wavesx.size());
	getperiods<T>(this->_wavesx, result.begin());
	return result;
}

template <class T>
arma::Array1D<T>
arma::stats::Wave_field<T>::lengths_y() const {
	Array1D<T> result(this->_wavesy.size());
	getperiods<T>(this->_wavesy, result.begin());
	return result;
}

template <class T>
arma::Array1D<T>
arma::stats::Wave_field<T>::heights() const {
	Array1D<T> result(_wavesx.size() + _wavesy.size());
	getheights<T>(_wavesy, getheights<T>(_wavesx, result.begin()));
	return result;
}

template <class T>
arma::Array1D<T>
arma::stats::Wave_field<T>::heights_x() const {
	Array1D<T> result(_wavesx.size());
	getheights<T>(_wavesx, result.begin());
	return result;
}

template <class T>
arma::Array1D<T>
arma::stats::Wave_field<T>::heights_y() const {
	Array1D<T> result(_wavesy.size());
	getheights<T>(_wavesy, result.begin());
	return result;
}

template <class T>
void
arma::stats::Wave_field<T>::extract_waves_t(
	Array3D<T> elevation,
	const Grid<T,3>& grid
) {
	using blitz::Range;
	const int nx = elevation.extent(1);
	const int ny = elevation.extent(2);
	const Grid<T,1> grid1d{{grid.num_points(0)}, {grid.length(0)}};
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < ny; ++j) {
			push_back_all(
				this->_wavest,
				factor_waves(find_extrema(
					elevation(Range::all(), i, j),
					grid1d
				))
			);
		}
	}
}

template <class T>
void
arma::stats::Wave_field<T>::extract_waves_x(
	Array3D<T> elevation,
	const Grid<T, 3>& grid
) {
	using blitz::Range;
	const int nt = elevation.extent(0);
	const int ny = elevation.extent(2);
	const Grid<T,1> grid1d{{grid.num_points(1)}, {grid.length(1)}};
	for (int i = 0; i < nt; ++i) {
		for (int j = 0; j < ny; ++j) {
			push_back_all(
				this->_wavesx,
				factor_waves(find_extrema(
					elevation(i, Range::all(), j),
					grid1d
				))
			);
		}
	}
}

template <class T>
void
arma::stats::Wave_field<T>::extract_waves_y(
	Array3D<T> elevation,
	const Grid<T, 3>& grid
) {
	using blitz::Range;
	const int nt = elevation.extent(0);
	const int nx = elevation.extent(1);
	const Grid<T,1> grid1d{{grid.num_points(2)}, {grid.length(2)}};
	for (int i = 0; i < nt; ++i) {
		for (int j = 0; j < nx; ++j) {
			push_back_all(
				this->_wavesy,
				factor_waves(find_extrema(
					elevation(i, j, Range::all()),
					grid1d
				))
			);
		}
	}
}

template <class T>
std::vector<arma::stats::Wave_feature<T>>
arma::stats::find_extrema(Array1D<T> elevation, const Grid<T,1>& grid) {
	std::vector<Wave_feature<T>> result;
	const int n = elevation.numElements();
	if (n < 3) {
		return result;
	}
	for (int i=1; i<n-1; ++i) {
		const T x0 = grid(i-1, 0);
		const T x1 = grid(i, 0);
		const T x2 = grid(i+1, 0);
		const T z0 = elevation(i-1);
		const T z1 = elevation(i);
		const T z2 = elevation(i+1);
		if ((z1 > z0 && z1 > z2) || (z1 < z0 && z1 < z2)) {
			// approximate three points with a parabola
			const T denominator = ((x0-x1) * (x0-x2) * (x1-x2));
			const T a = (x0*(z2-z1) + x1*(z0-z2) + x2*(z1-z0)) /
				denominator;
			const T b = (x0*x0*(z1-z2) + x1*x1*(z2-z0) + x2*x2*(z0-z1)) /
				denominator;
			const T c = (z0*(x1*x1*x2 - x1*x2*x2) +
					z1*(x0*x2*x2 - x0*x0*x2) +
					z2*(x0*x0*x1 - x0*x1*x1)) / denominator;
			// find parabola vertex
			const T vx = -b / (T(2)*a);
			const T vz = -(b*b - T(4)*a*c) / (T(4)*a);
			const Wave_feature_type type =
				z1 > z0 ? Wave_feature_type::Crest : Wave_feature_type::Trough;
			result.emplace_back(vx, vz, type);
		}
	}
	return result;
}

template <class T>
std::vector<arma::stats::Wave<T>>
arma::stats::factor_waves(const std::vector<Wave_feature<T>>& features) {
	using std::abs;
	std::vector<Wave<T>> result;
	if (features.empty()) {
		return result;
	}
	const int n = features.size();
	Wave_feature<T> f0 = features[0];
	for (int i=1; i<n; ++i) {
		Wave_feature<T> fi = features[i];
		if (fi.type != f0.type) {
			const T length = T(2)*abs(f0.x - fi.x);
			const T height = abs(f0.z - fi.z);
			result.emplace_back(height, length);
		}
		f0 = fi;
	}
	return result;
}

template <class T>
std::ostream&
arma::stats::operator<<(std::ostream& out, const Wave_feature<T>& rhs) {
	out << '(' << rhs.x << ',' << rhs.z << ',' << int(rhs.type) << ')';
	return out;
}

template class arma::stats::Wave<ARMA_REAL_TYPE>;
template class arma::stats::Wave_field<ARMA_REAL_TYPE>;

template std::vector<arma::stats::Wave_feature<ARMA_REAL_TYPE>>
arma::stats::find_extrema(
	Array1D<ARMA_REAL_TYPE> elevation,
	const Grid<ARMA_REAL_TYPE, 1>& grid
);

template std::vector<arma::stats::Wave<ARMA_REAL_TYPE>>
arma::stats::factor_waves(
	const std::vector<Wave_feature<ARMA_REAL_TYPE>>& features
);

template std::ostream&
arma::stats::operator<<(
	std::ostream& out,
	const Wave_feature<ARMA_REAL_TYPE>& rhs
);
