#ifndef ARMA_HH
#define ARMA_HH

#include <algorithm>  // for any_of, generate
#include <cassert>    // for assert
#include <chrono>     // for duration, steady_clock, steady_clock::...
#include <cmath>      // for isinf, isnan
#include <functional> // for bind
#include <random>     // for mt19937, normal_distribution
#include <stdexcept>  // for runtime_error
#include <ostream>
#include <fstream>
#include <type_traits>

#include <blitz/array.h>     // for Array
#include <gsl/gsl_complex.h> // for gsl_complex_packed_ptr
#include <gsl/gsl_errno.h>   // for gsl_strerror, ::GSL_SUCCESS
#include <gsl/gsl_poly.h>    // for gsl_poly_complex_solve, gsl_poly_com...

#include "types.hh" // for Zeta, ACF, size3
#include "statistics.hh"
#include "distribution.hh"
#include "fourier.hh"

/// @file
/// File with auxiliary subroutines.

namespace blitz {

	bool
	isfinite(float rhs) noexcept {
		return std::isfinite(rhs);
	}

	bool
	isfinite(double rhs) noexcept {
		return std::isfinite(rhs);
	}

	BZ_DECLARE_FUNCTION(isfinite);

	int
	div_ceil(int lhs, int rhs) noexcept {
		return lhs/rhs + (lhs%rhs == 0 ? 0 : 1);
	}

	BZ_DECLARE_FUNCTION2(div_ceil);

	template<int n>
	std::ostream&
	operator<<(std::ostream& out, const RectDomain<n>& rhs) {
		return out << rhs.lbound() << " : " << rhs.ubound();
	}
}

/// Domain-specific classes and functions.
namespace arma {

	/// Check AR (MA) process stationarity (invertibility).
	template <class T, int N>
	void
	validate_process(blitz::Array<T, N> _phi) {
		if (blitz::product(_phi.shape()) <= 1) { return; }
		/// 1. Find roots of the polynomial
		/// \f$P_n(\Phi)=1-\Phi_1 x-\Phi_2 x^2 - ... -\Phi_n x^n\f$.
		blitz::Array<double, N> phi(_phi.shape());
		phi = -_phi;
		phi(0) = 1;
		/// 2. Trim leading zero terms.
		size_t real_size = 0;
		while (real_size < phi.numElements() && phi.data()[real_size] != 0.0) {
			++real_size;
		}
		blitz::Array<std::complex<double>, 1> result(real_size);
		gsl_poly_complex_workspace* w =
		    gsl_poly_complex_workspace_alloc(real_size);
		int ret = gsl_poly_complex_solve(phi.data(), real_size, w,
		                                 (gsl_complex_packed_ptr)result.data());
		gsl_poly_complex_workspace_free(w);
		if (ret != GSL_SUCCESS) {
			std::clog << "GSL error: " << gsl_strerror(ret) << '.' << std::endl;
			throw std::runtime_error(
			    "Can not find roots of the polynomial to "
			    "verify AR/MA model stationarity/invertibility.");
		}
		/// 3. Check if some roots do not lie outside unit circle.
		size_t num_bad_roots = 0;
		for (size_t i = 0; i < result.size(); ++i) {
			const double val = std::abs(result(i));
			/// Some AR coefficients are close to nought and polynomial
			/// solver can produce noughts due to limited numerical
			/// precision. So we filter val=0 as well.
			if (!(val > 1.0 || val == 0.0)) {
				++num_bad_roots;
				std::clog << "Root #" << i << '=' << result(i) << std::endl;
			}
		}
		if (num_bad_roots > 0) {
			std::clog << "No. of bad roots = " << num_bad_roots << std::endl;
			throw std::runtime_error(
			    "AR/MA process is not stationary/invertible: some roots lie "
			    "inside unit circle or on its borderline.");
		}
	}

	template <class T>
	T
	ACF_variance(const ACF<T>& acf) {
		return acf(0, 0, 0);
	}

	/**
	Generate white noise via Mersenne Twister algorithm. Convert to normal
	distribution via Box---Muller transform.
	*/
	template <class T, class Generator>
	Zeta<T>
	generate_white_noise(const size3& size, T variance, Generator generator) {
		if (variance < T(0)) {
			throw std::runtime_error("variance is less than zero");
		}

		std::normal_distribution<T> normal(T(0), std::sqrt(variance));

		// generate and check
		Zeta<T> eps(size);
		std::generate(
			std::begin(eps),
			std::end(eps),
			std::bind(normal, generator)
		);
		if (!blitz::all(blitz::isfinite(eps))) {
			throw std::runtime_error(
			    "white noise generator produced some NaN/Inf");
		}
		return eps;
	}

	template <class T>
	struct Stats {

		template <int N, class D>
		Stats(blitz::Array<T, N> rhs, T m, T var, D dist, std::string name)
		    : _expected_mean(m), _mean(::stats::mean(rhs)),
		      _expected_variance(var), _variance(::stats::variance(rhs)),
		      _graph(dist, rhs), _name(name) {}

		T
		qdistance() const {
			return _graph.distance();
		}

		void
		write_quantile_graph() {
			std::string filename;
			std::transform(
			    _name.begin(), _name.end(), std::back_inserter(filename),
			    [](char ch) { return !std::isalnum(ch) ? '-' : ch; });
			std::ofstream out(filename);
			out << _graph;
		}

		friend std::ostream& operator<<(std::ostream& out, const Stats& rhs) {
			out.precision(5);
			out << std::setw(colw + 2) << rhs._name << std::setw(colw)
			    << rhs._mean << std::setw(colw) << rhs._variance
			    << std::setw(colw) << rhs._expected_mean << std::setw(colw)
			    << rhs._expected_variance << std::setw(colw) << rhs.qdistance();
			return out;
		}

		static void
		print_header(std::ostream& out) {
			out << std::setw(colw + 2) << "Property" << std::setw(colw)
			    << "Mean" << std::setw(colw) << "Var" << std::setw(colw)
			    << "ModelMean" << std::setw(colw) << "ModelVar"
			    << std::setw(colw) << "QDistance";
		}

	private:
		T _expected_mean;
		T _mean;
		T _expected_variance;
		T _variance;
		stats::Quantile_graph<T> _graph;
		std::string _name;

		static const int colw = 13;
	};

	template <class T, int N, class D>
	Stats<T>
	make_stats(blitz::Array<T, N> rhs, T m, T var, D dist, std::string name) {
		return Stats<T>(rhs, m, var, dist, name);
	}

	template <class T>
	struct Wave {

		Wave() = default;
		Wave(T height, T period) : _height(height), _period(period) {}

		T
		height() const {
			return _height;
		}

		T
		period() const {
			return _period;
		}

	private:
		T _height = 0;
		T _period = 0;
	};

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
				*result = Wave<T>(height, period * (n - 1));
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
				if (std::isnan(tex) || std::isnan(wex)) {
					std::clog << "NaN: " << tex << ", " << wex << ", " << a
					          << ", " << b << ", " << c << ", " << dw1 << ", "
					          << dw2 << std::endl;
				}
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
						*result = Wave<T>(height, period);
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

	template <class T>
	struct Wave_field {

		typedef std::vector<Wave<T>> wave_vector;

		explicit Wave_field(Array3D<T> elevation) {
			extract_waves_t(elevation);
			extract_waves_x(elevation);
			extract_waves_y(elevation);
			extract_waves_x2(elevation);
			extract_waves_y2(elevation);
		}

		Array1D<T>
		periods() {
			Array1D<T> result(_wavest.size());
			periods(_wavest, result.begin());
			return result;
		}

		Array1D<T>
		lengths() {
			Array1D<T> result(_wavesx2.size() + _wavesy2.size());
			periods(_wavesy2, periods(_wavesx2, result.begin()));
			return result;
		}

		Array1D<T>
		lengths_x() {
			Array1D<T> result(_wavesx2.size());
			periods(_wavesx2, result.begin());
			return result;
		}

		Array1D<T>
		lengths_y() {
			Array1D<T> result(_wavesy2.size());
			periods(_wavesy2, result.begin());
			return result;
		}

		Array1D<T>
		heights() {
			Array1D<T> result(_wavesx.size() + _wavesy.size());
			heights(_wavesy, heights(_wavesx, result.begin()));
			return result;
		}

		Array1D<T>
		heights_x() {
			Array1D<T> result(_wavesx.size());
			heights(_wavesx, result.begin());
			return result;
		}

		Array1D<T>
		heights_y() {
			Array1D<T> result(_wavesy.size());
			heights(_wavesy, result.begin());
			return result;
		}

	private:
		template <class Result>
		Result
		heights(const wave_vector& rhs, Result result) {
			return std::transform(
			    rhs.begin(), rhs.end(), result,
			    [](const Wave<T>& wave) { return wave.height(); });
		}

		template <class Result>
		Result
		periods(const wave_vector& rhs, Result result) {
			return std::transform(
			    rhs.begin(), rhs.end(), result,
			    [](const Wave<T>& wave) { return wave.period(); });
		}

		void
		extract_waves_t(Array3D<T> elevation) {
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

		void
		extract_waves_x(Array3D<T> elevation) {
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

		void
		extract_waves_y(Array3D<T> elevation) {
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

		void
		extract_waves_x2(Array3D<T> elevation) {
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

		void
		extract_waves_y2(Array3D<T> elevation) {
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

		wave_vector _wavest;
		wave_vector _wavesx;
		wave_vector _wavesy;
		wave_vector _wavesx2;
		wave_vector _wavesy2;
	};

	template <class T>
	T
	approx_wave_height(T variance) {
		return std::sqrt(T(2) * M_PI * variance);
	}

	template <class T>
	T
	approx_wave_period(T variance) {
		return T(4.8) * std::sqrt(approx_wave_height(variance));
	}

	template<int n>
	blitz::TinyVector<int, n>
	get_shape(const blitz::RectDomain<n>& rhs) {
		return rhs.ubound() - rhs.lbound() + 1;
	}

}

#endif // ARMA_HH
