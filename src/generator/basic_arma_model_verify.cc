#include "stats/statistics.hh"
#include "stats/distribution.hh"
#include "stats/summary.hh"
#include "stats/waves.hh"
#include "grid.hh"
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>

namespace {

	template <class T>
	void
	show_statistics(
		arma::Array3D<T> acf,
		arma::Array3D<T> zeta,
		const arma::generator::Basic_ARMA_model<T>& model,
		arma::Verification_scheme vscheme
	) {
		using namespace arma;
		using stats::Summary;
		using stats::Wave_field;
		const T var_wn = model.white_noise_variance();
		Summary<T>::print_header(std::clog);
		std::clog << std::endl;
		T var_elev = acf(0, 0, 0);
		Wave_field<T> wave_field(zeta);
		Array1D<T> heights_x = wave_field.heights_x();
		Array1D<T> heights_y = wave_field.heights_y();
		Array1D<T> periods = wave_field.periods();
		Array1D<T> lengths_x = wave_field.lengths_x();
		Array1D<T> lengths_y = wave_field.lengths_y();
		const T est_var_elev = stats::variance(zeta);
		stats::Gaussian<T> eps_dist(0, std::sqrt(var_wn));
		stats::Gaussian<T> elev_dist(0, std::sqrt(est_var_elev));
		stats::Wave_periods_dist<T> periods_dist(stats::mean(periods));
		stats::Wave_heights_dist<T> heights_x_dist(stats::mean(heights_x));
		stats::Wave_heights_dist<T> heights_y_dist(stats::mean(heights_y));
		stats::Wave_lengths_dist<T> lengths_x_dist(stats::mean(lengths_x));
		stats::Wave_lengths_dist<T> lengths_y_dist(stats::mean(lengths_y));
		std::vector<Summary<T>> stats = {
			make_summary(zeta, T(0), var_elev, elev_dist, "elevation"),
			make_summary(heights_x, approx_wave_height(var_elev), T(0),
					   heights_x_dist, "wave height x"),
			make_summary(heights_y, approx_wave_height(var_elev), T(0),
					   heights_y_dist, "wave height y"),
			make_summary(lengths_x, T(0), T(0), lengths_x_dist,
					   "wave length x"),
			make_summary(lengths_y, T(0), T(0), lengths_y_dist,
					   "wave length y"),
			make_summary(periods, approx_wave_period(var_elev), T(0),
					   periods_dist, "wave period"),
		};
		std::copy(
			stats.begin(),
			stats.end(),
			std::ostream_iterator<Summary<T>>(std::clog, "\n")
		);
		if (vscheme == Verification_scheme::Quantile) {
			std::for_each(
				stats.begin(),
				stats.end(),
				std::mem_fn(&Summary<T>::write_quantile_graph)
			);
		}
	}

	template<class T, int N>
	void
	write_raw(const char* filename, const blitz::Array<T,N>& x) {
		std::ofstream out(filename);
		std::copy(
			x.begin(),
			x.end(),
			std::ostream_iterator<T>(out, "\n")
		);
	}

	template <class T>
	void
	write_everything_to_files(arma::Array3D<T> acf, arma::Array3D<T> zeta) {
		arma::stats::Wave_field<T> wave_field(zeta);
		write_raw("heights_x", wave_field.heights_x());
		write_raw("heights_y", wave_field.heights_y());
		write_raw("periods", wave_field.periods());
		write_raw("lengths_x", wave_field.lengths_x());
		write_raw("lengths_y", wave_field.lengths_y());
		write_raw("elevation", zeta);
	}

}

template <class T>
void
arma::generator::Basic_ARMA_model<T>::verify(Array3D<T> zeta) const {
	switch (this->_vscheme) {
		case Verification_scheme::No_verification:
			break;
		case Verification_scheme::Summary:
		case Verification_scheme::Quantile:
			show_statistics(this->_acf, zeta, *this, this->_vscheme);
			break;
		case Verification_scheme::Manual:
			write_everything_to_files(this->_acf, zeta);
			break;
	}
}
