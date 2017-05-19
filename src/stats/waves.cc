#include "waves.hh"

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

}


template <class T>
arma::stats::Wave_field<T>::Wave_field(Array3D<T> elevation) {
	extract_waves_t(elevation);
	extract_waves_x(elevation);
	extract_waves_y(elevation);
	extract_waves_x2(elevation);
	extract_waves_y2(elevation);
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
	Array1D<T> result(_wavesx2.size() + _wavesy2.size());
	getperiods<T>(_wavesy2, getperiods<T>(_wavesx2, result.begin()));
	return result;
}

template <class T>
arma::Array1D<T>
arma::stats::Wave_field<T>::lengths_x() const {
	Array1D<T> result(_wavesx2.size());
	getperiods<T>(_wavesx2, result.begin());
	return result;
}

template <class T>
arma::Array1D<T>
arma::stats::Wave_field<T>::lengths_y() const {
	Array1D<T> result(_wavesy2.size());
	getperiods<T>(_wavesy2, result.begin());
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
arma::stats::Wave_field<T>::extract_waves_t(Array3D<T> elevation) {
	using blitz::Range;
	const int nx = elevation.extent(1);
	const int ny = elevation.extent(2);
	auto ins = std::back_inserter(_wavest);
	for (int i = 0; i < nx; ++i) {
		for (int j = 0; j < ny; ++j) {
			Array1D<T> elev1d = elevation(Range::all(), i, j);
			copy_waves_t(elev1d.data(), elev1d.numElements(), ins);
		}
	}
}

template <class T>
void
arma::stats::Wave_field<T>::extract_waves_x(Array3D<T> elevation) {
	using blitz::Range;
	const int nt = elevation.extent(0);
	const int ny = elevation.extent(2);
	auto ins = std::back_inserter(_wavesx);
	for (int i = 0; i < nt; ++i) {
		for (int j = 0; j < ny; ++j) {
			Array1D<T> elev1d = elevation(i, Range::all(), j);
			copy_waves_x(elev1d.data(), elev1d.numElements(), ins);
		}
	}
}

template <class T>
void
arma::stats::Wave_field<T>::extract_waves_y(Array3D<T> elevation) {
	using blitz::Range;
	const int nt = elevation.extent(0);
	const int nx = elevation.extent(1);
	auto ins = std::back_inserter(_wavesy);
	for (int i = 0; i < nt; ++i) {
		for (int j = 0; j < nx; ++j) {
			Array1D<T> elev1d = elevation(i, j, Range::all());
			copy_waves_x(elev1d.data(), elev1d.numElements(), ins);
		}
	}
}

template <class T>
void
arma::stats::Wave_field<T>::extract_waves_x2(Array3D<T> elevation) {
	using blitz::Range;
	const int nt = elevation.extent(0);
	const int ny = elevation.extent(2);
	auto ins = std::back_inserter(_wavesx2);
	for (int i = 0; i < nt; ++i) {
		for (int j = 0; j < ny; ++j) {
			Array1D<T> elev1d = elevation(i, Range::all(), j);
			copy_waves_t(elev1d.data(), elev1d.numElements(), ins);
		}
	}
}

template <class T>
void
arma::stats::Wave_field<T>::extract_waves_y2(Array3D<T> elevation) {
	using blitz::Range;
	const int nt = elevation.extent(0);
	const int nx = elevation.extent(1);
	auto ins = std::back_inserter(_wavesy2);
	for (int i = 0; i < nt; ++i) {
		for (int j = 0; j < nx; ++j) {
			Array1D<T> elev1d = elevation(i, j, Range::all());
			copy_waves_t(elev1d.data(), elev1d.numElements(), ins);
		}
	}
}


template class arma::stats::Wave<ARMA_REAL_TYPE>;
template class arma::stats::Wave_field<ARMA_REAL_TYPE>;
