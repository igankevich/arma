#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "grid.hh"
#include "stats/distribution.hh"
#include "stats/statistics.hh"
#include "stats/summary.hh"
#include "stats/waves.hh"
#include "util.hh"

namespace {

	template <class T>
	void
	print_indicator(
		std::string prefix,
		std::string name,
		T expected,
		T actual,
		bool ok
	) {
		std::clog
			<< std::left
			<< std::setw(15) << prefix
			<< std::setw(20) << name
			<< std::right
			<< std::setw(10) << expected
			<< std::setw(10) << actual
			<< std::setw(10) << (ok ? "ok" : "wrong")
			<< std::endl;
	}

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
		const T var_wn = model.white_noise_variance();
		Array3D<T> spectrum =
			arma::stats::frequency_amplitude_spectrum(zeta, model.grid());
		{
			std::ofstream out("spectrum");
			out << spectrum;
		}
		T var_elev = acf(0,0,0);
		const T r = 5;
		Wave_field<T> wave_field(zeta, model.grid(), r);
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
			make_summary(
				zeta,
				T(0),
				var_elev,
				elev_dist,
				"elevation",
				T(0.1)
			),
			make_summary(
				periods,
				model.acf_generator().wave_period(),
				periods_dist,
				"wave period",
				r*model.grid().delta(0)
			),
		};
		if (model.acf_generator().has_x()) {
			stats.emplace_back(
				heights_x,
				model.acf_generator().wave_height_x(),
				T(0),
				heights_x_dist,
				"wave height x",
				false,
				T(0.5)
			);
			stats.emplace_back(
				lengths_x,
				model.acf_generator().wave_length_x(),
				T(0),
				lengths_x_dist,
				"wave length x",
				false,
				r*model.grid().delta(1)
			);
		}
		if (model.acf_generator().has_y()) {
			stats.emplace_back(
				heights_y,
				model.acf_generator().wave_height_y(),
				T(0),
				heights_y_dist,
				"wave height y",
				false,
				T(0.5)
			);
			stats.emplace_back(
				lengths_y,
				model.acf_generator().wave_length_y(),
				T(0),
				lengths_y_dist,
				"wave length y",
				false,
				r*model.grid().delta(2)
			);
		}
		/*
		if (oflags.isset(Output_flags::Summary)) {
			Summary<T>::print_header(std::clog);
			std::clog << std::endl;
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
			std::clog
				<< "Min/Max wave height x: "
				<< blitz::min(heights_x)
				<< " / "
				<< blitz::max(heights_x)
				<< std::endl;
			std::clog
				<< "Min/Max wave height y: "
				<< blitz::min(heights_y)
				<< " / "
				<< blitz::max(heights_y)
				<< std::endl;
			std::clog
				<< "Min/Max wave length x: "
				<< blitz::min(lengths_x)
				<< " / "
				<< blitz::max(lengths_x)
				<< std::endl;
			std::clog
				<< "Min/Max wave length y: "
				<< blitz::min(lengths_y)
				<< " / "
				<< blitz::max(lengths_y)
				<< std::endl;
			std::clog
				<< "Average amplitude: "
				<< blitz::max(spectrum)
				<< std::endl;
			std::clog
				<< "Average wave number x: "
				<< Vec3D<T>(
					blitz::max_element(spectrum).position()*model.grid().delta()
				)
				<< std::endl;
			std::clog
				<< "Min/Max z: "
				<< blitz::min(zeta)
				<< " / "
				<< blitz::max(zeta)
				<< std::endl;
		}
		*/
		if (oflags.isset(Output_flags::Quantile)) {
			std::for_each(
				stats.begin(),
				stats.end(),
				std::mem_fn(&Summary<T>::write_quantile_graph)
			);
		}
		if (oflags.isset(Output_flags::Summary)) {
			std::clog << "No. of waves = "
				<< Shape3D(periods.size(), lengths_x.size(), lengths_y.size())
				<< std::endl;
			auto oldf = std::clog.flags();
			std::clog.setf(std::ios::fixed, std::ios::floatfield);
			std::clog.setf(std::ios::boolalpha);
			std::clog.precision(3);
			for (const Summary<T>& s : stats) {
				print_indicator(
					"mean",
					s.name(),
					s.expected_mean(),
					s.mean(),
					s.mean_ok()
				);
				if (s.has_variance()) {
					print_indicator(
						"variance",
						s.name(),
						s.expected_variance(),
						s.variance(),
						s.variance_ok()
					);
				}
			}
			for (const Summary<T>& s : stats) {
				print_indicator(
					"qdistance",
					s.name(),
					T(0),
					s.qdistance(),
					s.qdistance_ok()
				);
			}

			if (model.acf_generator().has_x()) {
				const T r = stats::mean(lengths_x) / stats::mean(heights_x);
				print_indicator(
					"ratio",
					"height to length x",
					T(7),
					r,
					r > T(7) && r < T(40)
				);
			}
			if (model.acf_generator().has_y()) {
				const T r = stats::mean(lengths_y) / stats::mean(heights_y);
				print_indicator(
					"ratio",
					"height to length y",
					T(7),
					r,
					r > T(7) && r < T(40)
				);
			}
			std::clog.setf(oldf);
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
		arma::stats::Wave_field<T> wave_field(zeta, grid, 11);
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
		/*
		{
			using blitz::Range;
			std::ofstream out("slice_tx");
			const int ni = zeta.extent(0);
			const int nj = zeta.extent(1);
			const int k = zeta.extent(2)/2;
			Array3D<T> slice(zeta(
				Range::all(),
				Range::all(),
				Range(k, k)
			));
			for (int i=0; i<ni; ++i) {
				Array1D<T> slice_x(slice(i, Range::all()));
				Domain<T,1> subdomain{{T(0)}, {this->_outgrid.ubound(1)}, {nj}};
				arma::stats::smooth_elevation(slice_x, subdomain, 11);
				slice(i, Range::all()) = slice_x;
			}
			out << slice;
		}
		*/
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
