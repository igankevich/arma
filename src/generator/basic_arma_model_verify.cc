#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>

#include "grid.hh"
#include "stats/distribution.hh"
#include "stats/statistics.hh"
#include "stats/summary.hh"
#include "stats/waves.hh"
#include "util.hh"

namespace {

	template <class T>
	void
	show_statistics(
		arma::Array3D<T> acf,
		arma::Array3D<T> zeta,
		const arma::generator::Basic_ARMA_model<T>& model,
		arma::Output_flags oflags
	) {
		using namespace arma;
		using stats::Summary;
		using stats::Wave_field;
		Array1D<T> slice_x(zeta(
			zeta.extent(0)-1,
			blitz::Range::all(),
			zeta.extent(2)-1
		));
		std::clog << "slice_x=" << slice_x << std::endl;
		const T var_wn = model.white_noise_variance();
		Summary<T>::print_header(std::clog);
		std::clog << std::endl;
		T var_elev = acf(0, 0, 0);
		Wave_field<T> wave_field(zeta, model.grid());
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
			make_summary(
				heights_x,
				model.acf_generator().wave_height(),
				heights_x_dist,
				"wave height x"
			),
			make_summary(
				heights_y,
				model.acf_generator().wave_height(),
				heights_y_dist,
				"wave height y"
			),
			make_summary(
				lengths_x,
				model.acf_generator().wave_length_x(),
				lengths_x_dist,
				"wave length x"
			),
			make_summary(
				lengths_y,
				model.acf_generator().wave_length_y(),
				lengths_y_dist,
				"wave length y"
			),
			make_summary(
				periods,
				model.acf_generator().wave_period(),
				periods_dist,
				"wave period"
			),
		};
		if (oflags.isset(Output_flags::Summary)) {
			std::copy(
				stats.begin(),
				stats.end(),
				std::ostream_iterator<Summary<T>>(std::clog, "\n")
			);
			std::clog
				<< "Wave height to length ratio x: 1 to "
				<< (stats::mean(lengths_x) / stats::mean(heights_x))
				<< std::endl;
			std::clog
				<< "Wave height to length ratio y: 1 to "
				<< (stats::mean(lengths_y) / stats::mean(heights_y))
				<< std::endl;
		}
		if (oflags.isset(Output_flags::Quantile)) {
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
	write_everything_to_files(
		arma::Array3D<T> acf,
		arma::Array3D<T> zeta,
		const arma::Grid<T,3>& grid
	) {
		arma::stats::Wave_field<T> wave_field(zeta, grid);
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
	using blitz::RectDomain;
	if (this->_oflags.isset(Output_flags::Summary) ||
		this->_oflags.isset(Output_flags::Quantile))
	{
		show_statistics(
			this->_acf,
			zeta(RectDomain<3>(zeta.shape()/2, zeta.shape()-1)),
			*this,
			this->_oflags
		);
	}
	if (this->_oflags.isset(Output_flags::Waves)) {
		write_everything_to_files(this->_acf, zeta, this->_outgrid);
	}
}
