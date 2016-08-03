#ifndef AUTOREG_HH
#define AUTOREG_HH

#include <algorithm>  // for any_of, generate
#include <cassert>    // for assert
#include <chrono>     // for duration, steady_clock, steady_clock::...
#include <cmath>      // for isinf, isnan
#include <functional> // for bind
#include <random>     // for mt19937, normal_distribution
#include <stdexcept>  // for runtime_error
#include <ostream>

#include <blitz/array.h>     // for Array
#include <gsl/gsl_complex.h> // for gsl_complex_packed_ptr
#include <gsl/gsl_errno.h>   // for gsl_strerror, ::GSL_SUCCESS
#include <gsl/gsl_poly.h>    // for gsl_poly_complex_solve, gsl_poly_com...

#include "types.hh" // for Zeta, ACF, size3
#include "statistics.hh"
#include "distribution.hh"

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
}

/// Domain-specific classes and functions.
namespace autoreg {

	/// Check AR (MA) process stationarity (invertibility).
	template <class T, int N>
	void
	validate_process(blitz::Array<T, N> _phi) {
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
	template <class T>
	Zeta<T>
	generate_white_noise(const size3& size, T variance) {
		if (variance < T(0)) {
			throw std::runtime_error("variance is less than zero");
		}

		// initialise generator
		std::mt19937 generator;
#if !defined(DISABLE_RANDOM_SEED)
		generator.seed(
		    std::chrono::steady_clock::now().time_since_epoch().count());
#endif
		std::normal_distribution<T> normal(T(0), std::sqrt(variance));

		// generate and check
		Zeta<T> eps(size);
		std::generate(std::begin(eps), std::end(eps),
		              std::bind(normal, generator));
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
		      _distance(dist.distance(rhs)), _name(name) {}

		friend std::ostream& operator<<(std::ostream& out, const Stats& rhs) {
			out.precision(5);
			out << std::setw(colw+2) << rhs._name << std::setw(colw) << rhs._mean
			    << std::setw(colw) << rhs._variance << std::setw(colw)
			    << rhs._expected_mean << std::setw(colw)
			    << rhs._expected_variance << std::setw(colw) << rhs._distance;
			return out;
		}

		static void
		print_header(std::ostream& out) {
			out << std::setw(colw+2) << "Property" << std::setw(colw) << "Mean"
			    << std::setw(colw) << "Var" << std::setw(colw) << "ModelMean"
			    << std::setw(colw) << "ModelVar" << std::setw(colw)
			    << "QDistance";
		}

	private:
		T _expected_mean;
		T _mean;
		T _expected_variance;
		T _variance;
		T _distance;
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

	template <class It, class Result>
	void
	copy_waves(It elevation, size_t n, Result result) {
		typedef decltype(*elevation) T;
		int trough_first = -1;
		int crest = -1;
		int trough_last = -1;
		T elev0 = *elevation;
		++elevation;
		T elev1 = *elevation;
		++elevation;
		T elev_trough_first, elev_trough_last, elev_crest;
		for (size_t j = 2; j < n; ++j) {
			T elev2 = *elevation;
			if (elev0 < elev1 && elev2 < elev1) {
				crest = j - 1;
				elev_crest = elev1;
			} else if (elev1 < elev0 && elev1 < elev2) {
				if (trough_first == -1) {
					trough_first = j - 1;
					elev_trough_first = elev1;
				} else {
					trough_last = j - 1;
					elev_trough_last = elev1;
				}
			}
			if (trough_first != -1 && crest != -1 && trough_last != -1) {
				const T height =
				    (T(2) * elev_crest - elev_trough_first - elev_trough_last) *
				    T(0.5);
				const T period = trough_last - trough_first;
				*result = Wave<T>(height, period);
				++result;
				trough_first = trough_last;
				crest = -1;
				trough_last = -1;
			}
			elev0 = elev1;
			elev1 = elev2;
			++elevation;
		}
	}

	template <class T>
	struct Wave_field {

		typedef std::vector<Wave<T>> wave_vector;

		explicit Wave_field(Array3D<T> elevation) {
			extract_waves_t(elevation);
			extract_waves_x(elevation);
			extract_waves_y(elevation);
		}

		Array1D<T>
		periods() {
			Array1D<T> result(_wavest.size());
			periods(_wavest, result.begin());
			return result;
		}

		Array1D<T>
		lengths() {
			Array1D<T> result(_wavesx.size() + _wavesy.size());
			periods(_wavesy, periods(_wavesx, result.begin()));
			return result;
		}

		Array1D<T>
		heights() {
			Array1D<T> result(_wavesx.size() + _wavesy.size());
			heights(_wavesy, heights(_wavesx, result.begin()));
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
					copy_waves(elev1d.begin(), elev1d.numElements(), ins);
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
					copy_waves(elev1d.begin(), elev1d.numElements(), ins);
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
					copy_waves(elev1d.begin(), elev1d.numElements(), ins);
				}
			}
		}

		wave_vector _wavest;
		wave_vector _wavesx;
		wave_vector _wavesy;
	};
}

#endif // AUTOREG_HH
