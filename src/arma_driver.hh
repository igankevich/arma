#ifndef ARMA_DRIVER_HH
#define ARMA_DRIVER_HH

#include <iostream>  // for operator<<, basic_ostream, clog
#include <iomanip>   // for operator<<, setw
#include <fstream>   // for ofstream
#include <stdexcept> // for runtime_error
#include <string>    // for operator==, basic_string, string, getline
#include <unordered_map>
#include <functional>

#include "types.hh"   // for size3, Vector, Zeta, ACF, AR_coefs
#include "arma.hh"    // for mean, variance, ACF_variance, approx_acf, comp...
#include "ar_model.hh"
#include "ma_model.hh"
#include "arma_model.hh"
#include "simulation_model.hh"
#include "verification_scheme.hh" // for Verification_scheme
#include "acf.hh" // for standing_wave_ACF, propagating_wave_ACF
#include "params.hh"
#include "grid.hh"
#include "statistics.hh"
#include "distribution.hh"

/// @file
/// Some abbreviations used throughout the programme.
/// AR      autoregressive
/// ACF     auto-covariate function
/// zeta    ocean wavy surface
/// phi     AR model coefficients
/// YW      Yule-Walker
/// WN      white noise
/// var     variance
/// MT      Mersenne Twister (pseudo-random number generator)

namespace arma {

	template <class T>
	void
	write_key_value(std::ostream& out, const char* key, T value) {
		std::ios::fmtflags oldf =
		    out.setf(std::ios::left, std::ios::adjustfield);
		out << std::setw(30) << key << " = " << value << std::endl;
		out.setf(oldf);
	}

	/// Class that reads parameters from the input files,
	/// calls all subroutines, and prints the result.
	template <class T>
	struct Autoreg_model {

		typedef std::function<ACF<T>(const Vec3<T>&, const size3&)>
		    ACF_function;

		typedef std::chrono::high_resolution_clock clock_type;

		Autoreg_model()
		    : _outgrid{{768, 24, 24}},
		      _acfgrid{{10, 10, 10}, {T(2.5), T(5), T(5)}},
		      _arorder(_acfgrid.size()) {}

		void
		act() {
			echo_parameters();
			ACF_function acf_func = get_acf_function();
			ACF<T> acf = acf_func(_acfgrid.delta(), _acfgrid.size());
			std::clog << "ACF variance = " << ACF_variance(acf) << std::endl;
			if (_vscheme == Verification_scheme::Manual) {
				write_csv("acf.csv", acf);
			}
			{
				std::ofstream out("acf");
				out << acf;
			}
			{
				std::ofstream out("zdelta");
				out << _acfgrid.delta();
			}
			Zeta<T> eps(_outgrid.size());
			Zeta<T> zeta(_outgrid.size());
			#if ARMA_FRAMEWORK != NONE
			parallel_mt prng;
			std::vector<arma::mt_config> prng_config;
			read_parallel_mt_config(
				"parallel_mt.dat",
				std::back_inserter(prng_config)
			);
			#else
			std::mt19937 prng;
			#if !defined(ARMA_NO_PRNG_SEED)
			prng.seed(clock_type::now().time_since_epoch().count());
			#endif
			#endif
			if (_model == Simulation_model::Autoregressive) {
				Autoregressive_model<T> model(acf, _arorder);
				model.determine_coefficients(_doleastsquares);
				model.validate();
				T var_wn = model.white_noise_variance();
				std::clog << "WN variance = " << var_wn << std::endl;
				eps = generate_white_noise(_outgrid.size(), var_wn, std::ref(prng));
				Zeta<T> tmp = eps.copy();
				tmp = model(tmp);
				write_zeta(tmp);
				/// Estimate mean/variance with ramp-up region removed.
				blitz::RectDomain<3> subdomain(_arorder, tmp.shape() - 1);
				zeta.resize(subdomain[0], subdomain[1], subdomain[2]);
				std::clog << "Zeta size = " << zeta.shape() << std::endl;
				zeta = tmp(subdomain);
//				zeta = tmp;
				verify(acf, eps, zeta, model);
			} else if (_model == Simulation_model::Moving_average) {
				Moving_average_model<T> model(acf, _maorder);
				model.determine_coefficients(1000, T(1e-5), T(1e-6),
				                             _ma_algorithm);
				model.validate();
				T var_wn = model.white_noise_variance();
				std::clog << "WN variance = " << var_wn << std::endl;
				eps = generate_white_noise(_outgrid.size(), var_wn, std::ref(prng));
				zeta = model(eps);
				write_zeta(zeta);
				verify(acf, eps, zeta, model);
			} else if (_model == Simulation_model::ARMA) {
				ARMA_model<T> model(acf, _arorder, _maorder);
				model.determine_coefficients();
				model.validate();
				T var_wn = model.white_noise_variance();
				std::clog << "WN variance = " << var_wn << std::endl;
				eps = generate_white_noise(_outgrid.size(), var_wn, std::ref(prng));
				zeta = model(eps);
				write_zeta(zeta);
				verify(acf, eps, zeta, model);
			}
		}

		/**
		Read AR model parameters from an input stream, generate default ACF
		and validate all the parameters.
		*/
		template <class V>
		friend std::istream&
		operator>>(std::istream& in, Autoreg_model<V>& m) {
			m.read_parameters(in);
			m.validate_parameters();
			return in;
		}

	private:
		template <class Model>
		void
		verify(ACF<T> acf, Zeta<T> eps, Zeta<T> zeta, Model model) {
			switch (_vscheme) {
				case Verification_scheme::None:
					break;
				case Verification_scheme::Summary:
				case Verification_scheme::Quantile:
					show_statistics(acf, eps, zeta, model);
					break;
				case Verification_scheme::Manual:
					write_everything_to_files(acf, eps, zeta);
					break;
			}
		}

		template<class Model>
		void
		show_statistics(ACF<T> acf, Zeta<T> eps, Zeta<T> zeta, Model model) {
			const T var_wn = model.white_noise_variance();
			Stats<T>::print_header(std::clog);
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
			std::vector<Stats<T>> stats = {
			    make_stats(eps, T(0), var_wn, eps_dist, "white noise"),
			    make_stats(zeta, T(0), var_elev, elev_dist, "elevation"),
			    make_stats(heights_x, approx_wave_height(var_elev), T(0),
			               heights_x_dist, "wave height x"),
			    make_stats(heights_y, approx_wave_height(var_elev), T(0),
			               heights_y_dist, "wave height y"),
			    make_stats(lengths_x, T(0), T(0), lengths_x_dist,
			               "wave length x"),
			    make_stats(lengths_y, T(0), T(0), lengths_y_dist,
			               "wave length y"),
			    make_stats(periods, approx_wave_period(var_elev), T(0),
			               periods_dist, "wave period"),
			};
			std::copy(
				stats.begin(),
				stats.end(),
				std::ostream_iterator<Stats<T>>(std::clog, "\n")
			);
			if (_vscheme == Verification_scheme::Quantile) {
				std::for_each(
					stats.begin(),
					stats.end(),
					std::mem_fn(&Stats<T>::write_quantile_graph)
				);
			}
		}

		void
		write_everything_to_files(ACF<T> acf, Zeta<T> eps, Zeta<T> zeta) {
			Wave_field<T> wave_field(zeta);
			write_raw("heights_x", wave_field.heights_x());
			write_raw("heights_y", wave_field.heights_y());
			write_raw("periods", wave_field.periods());
			write_raw("lengths_x", wave_field.lengths_x());
			write_raw("lengths_y", wave_field.lengths_y());
			write_raw("elevation", zeta);
			write_raw("white_noise", eps);
		}

		/// Read AR model parameters from an input stream.
		void
		read_parameters(std::istream& in) {
			sys::parameter_map params({
			    {"out_grid", sys::make_param(_outgrid)},
			    {"acf_grid", sys::make_param(_acfgrid)},
			    {"ar_order", sys::make_param(_arorder)},
			    {"ma_order", sys::make_param(_maorder)},
			    {"least_squares", sys::make_param(_doleastsquares)},
			    {"acf", sys::make_param(_acffunc)},
			    {"model", sys::make_param(_model)},
			    {"ma_algorithm", sys::make_param(_ma_algorithm)},
			    {"verification", sys::make_param(_vscheme)},
			});
			in >> params;
		}

		/// Check for common input/logical errors and numerical implementation
		/// constraints.
		void
		validate_parameters() {
			check_non_zero(_outgrid.size(), "output grid size");
			check_non_zero(_outgrid.delta(), "output grid patch size");
			check_non_zero(_acfgrid.size(), "ACF grid size");
			check_non_zero(_acfgrid.delta(), "ACF grid patch size");
			int part_sz = _outgrid.num_points(0);
			int fsize_t = _arorder[0];
			if (fsize_t > part_sz) {
				std::stringstream tmp;
				tmp << "_arorder[0] > zsize[0], should be 0 < _arorder[0] < "
				       "zsize[0]\n";
				tmp << "_arorder[0]  = " << fsize_t << '\n';
				tmp << "zsize[0] = " << part_sz << '\n';
				throw std::runtime_error(tmp.str());
			}
		}

		/// Check that all components of vector @sz are non-zero,
		/// i.e. it is valid size specification.
		template <class V, int N>
		void
		check_non_zero(const Vector<V, N>& sz, const char* var_name) {
			if (blitz::product(sz) == V(0)) {
				std::stringstream str;
				str << "Invalid " << var_name << ": " << sz;
				throw std::runtime_error(str.str().c_str());
			}
		}

		void
		echo_parameters() {
			write_key_value(std::clog, "ACF grid size", _acfgrid.size());
			write_key_value(std::clog, "ACF grid patch size",
			                _acfgrid.patch_size());
			write_key_value(std::clog, "Output grid size", _outgrid.size());
			write_key_value(std::clog, "Output grid patch size",
			                _outgrid.patch_size());
			write_key_value(std::clog, "AR order", _arorder);
			write_key_value(std::clog, "Do least squares", _doleastsquares);
			write_key_value(std::clog, "ACF function", _acffunc);
			write_key_value(std::clog, "Model", _model);
			write_key_value(std::clog, "MA algorithm", _ma_algorithm);
			write_key_value(std::clog, "Verification scheme", _vscheme);
		}

		void
		write_zeta(const Zeta<T>& zeta) {
			std::ofstream out("zeta");
			out << zeta;
			if (_vscheme == Verification_scheme::Manual) {
				write_csv("zeta.csv", zeta);
			}
		}

		template<class X, int N>
		void
		write_raw(const char* filename, const blitz::Array<X,N>& x) {
			std::ofstream out(filename);
			std::copy(
				x.begin(),
				x.end(),
				std::ostream_iterator<T>(out, "\n")
			);
		}

		template<class X>
		void
		write_csv(
			const char* filename,
			const blitz::Array<X, 3>& data,
			const char separator=','
		) {
			std::ofstream out(filename);
			out << 't' << separator
			   << 'x' << separator
			   << 'y' << separator
			   << 'z' << '\n';
			const int nt = data.extent(0);
			const int nx = data.extent(1);
			const int ny = data.extent(2);
			for (int i=0; i<nt; ++i) {
				for (int j=0; j<nx; ++j) {
					for (int k=0; k<ny; ++k) {
						out << i << separator
							<< j << separator
							<< k << separator
							<< data(i, j, k)
							<< '\n';
					}
				}
			}
		}

		#if ARMA_FRAMEWORK != NONE
		void
		write_parallel_mt_config(const char* filename) {
			std::ofstream out(filename);
			if (!in.is_open()) {
				throw std::runtime_error("bad file");
			}
			std::ostream_iterator<arma::mt_config> out_it(out);
			arma::parallel_mt_seq<> seq(seed);
			std::generate_n(out_it, n, std::ref(seq));
		}

		template<class Result>
		void
		read_parallel_mt_config(const char* filename, Result result) {
			std::ifstream in(filename);
			if (!in.is_open()) {
				write_parallel_mt_config(filename);
				in.open(filename);
			}
			if (!in.is_open()) {
				throw std::runtime_error("bad file");
			}
			std::copy(
				std::istream_iterator<arma::mt_config>(in),
				std::istream_iterator<arma::mt_config>(),
				result
			);
		}
		#endif

		ACF_function
		get_acf_function() {
			auto result = acf_functions.find(_acffunc);
			if (result == acf_functions.end()) {
				std::clog << "Invalid ACF function name: \"" << _acffunc << '\"'
				          << std::endl;
				throw std::runtime_error("bad ACF function name");
			}
			return result->second;
		}

		Grid<T, 3> _outgrid; //< Wavy surface grid.
		Grid<T, 3> _acfgrid; //< ACF grid.
		size3 _arorder;      //< AR model order (no. of coefficients).
		size3 _maorder;      //< MA model order (no. of coefficients).
		bool _doleastsquares = false;

		/// ACF function name (\see acf_functions). Default is "standing_wave".
		std::string _acffunc = "standing_wave";

		Simulation_model _model = Simulation_model::Autoregressive;
		MA_algorithm _ma_algorithm = MA_algorithm::Fixed_point_iteration;
		Verification_scheme _vscheme = Verification_scheme::Summary;

		/// Map of names to ACF functions.
		static const std::unordered_map<std::string, ACF_function>
		    acf_functions;
	};

	template <class T>
	const std::unordered_map<
	    std::string, std::function<ACF<T>(const Vec3<T>&, const size3&)>>
	    Autoreg_model<T>::acf_functions = {
	        {"standing_wave", standing_wave_ACF<T>},
	        {"propagating_wave", propagating_wave_ACF<T>}};
}

#endif // ARMA_DRIVER_HH
