#ifndef ARMA_DRIVER_HH
#define ARMA_DRIVER_HH

#include <iostream>  // for operator<<, basic_ostream, clog
#include <iomanip>   // for operator<<, setw
#include <fstream>   // for ofstream
#include <stdexcept> // for runtime_error
#include <string>    // for operator==, basic_string, string, getline
#include <unordered_map>
#include <functional>
#include <type_traits>
#include <iterator>
#if ARMA_OPENMP
#include <omp.h>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <cmath>
#endif

#include "config.hh"
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
#include "parallel_mt.hh"
#include "errors.hh"

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
			struct {
				bool least_squares;
				int max_iterations;
			   	T eps;
			   	T min_var_wn;
				MA_algorithm algo;
			} opts;
			opts.max_iterations = 1000;
			opts.eps = T(1e-5);
			opts.min_var_wn = T(1e-6);
			opts.algo = _ma_algorithm;
			opts.least_squares = _doleastsquares;
			if (_model == Simulation_model::Autoregressive) {
				Autoregressive_model<T> model(acf, _arorder);
				generate_wavy_surface(model, opts);
			} else if (_model == Simulation_model::Moving_average) {
				Moving_average_model<T> model(acf, _maorder);
				generate_wavy_surface(model, opts);
			} else if (_model == Simulation_model::ARMA) {
				ARMA_model<T> model(acf, _arorder, _maorder);
				generate_wavy_surface(model, opts);
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
		template <class Model, class Options>
		void
		generate_wavy_surface(Model& model, const Options& opts) {
			model.determine_coefficients(opts);
			model.validate();
			T var_wn = model.white_noise_variance();
			std::clog << "WN variance = " << var_wn << std::endl;
			Zeta<T> zeta = do_generate_wavy_surface(model, opts, var_wn);
			write_zeta(zeta);
			if (std::is_same<Model,Autoregressive_model<T>>::value) {
				/// Estimate mean/variance with ramp-up region removed.
				blitz::RectDomain<3> subdomain(model.order(), zeta.shape() - 1);
				size3 zeta_size = subdomain.ubound() - subdomain.lbound();
				std::clog << "Zeta size = " << zeta_size << std::endl;
				verify(model.acf(), zeta(subdomain), model);
			} else {
				verify(model.acf(), zeta, model);
			}
		}

		#if ARMA_NONE

		template <class Model, class Options>
		Zeta<T>
		do_generate_wavy_surface(
			Model& model,
			const Options& opts,
			T var_wn
		) {
			std::mt19937 prng;
			prng.seed(newseed());
			Zeta<T> eps = generate_white_noise(
				_outgrid.size(),
				var_wn,
				std::ref(prng)
			);
			Zeta<T> zeta(eps.shape());
			model(zeta, eps);
			return zeta;
		}

		#elif ARMA_OPENMP

		struct Partition {

			Partition() = default;

			Partition(size3 ijk_, const blitz::RectDomain<3>& r, const mt_config& conf):
			ijk(ijk_), rect(r), prng(conf)
			{}

			friend std::ostream&
			operator<<(std::ostream& out, const Partition& rhs) {
				return out << rhs.ijk << ": " << rhs.rect;
			}

			size3
			shape() const {
				return get_shape(rect);
			}

			size3 ijk;
			blitz::RectDomain<3> rect;
			parallel_mt prng;
		};

		template <class Model, class Options>
		Zeta<T>
		do_generate_wavy_surface(
			Model& model,
			const Options& opts,
			T var_wn
		) {
			using blitz::RectDomain;
			/// 1. Read parallel Mersenne Twister states.
			std::vector<mt_config> prng_config;
			read_parallel_mt_config(
				config::mt_config_file,
				std::back_inserter(prng_config)
			);
			const int nprngs = prng_config.size();
			if (nprngs == 0) {
				throw prng_error("bad number of MT configs", nprngs, 0);
			}
			/// 2. Partition the data.
			const size3 shape = _outgrid.size();
			const size3 partshape = get_partition_shape(model.order(), nprngs);
			const size3 nparts = blitz::div_ceil(shape, partshape);
			const int ntotal = blitz::product(nparts);
			if (prng_config.size() < size_t(blitz::product(nparts))) {
				throw prng_error("bad number of MT configs", nprngs, ntotal);
			}
			write_key_value(std::clog, "Partition size", partshape);
			std::vector<Partition> parts = partition(
				nparts,
				partshape,
				shape,
				prng_config
			);
			Array3D<bool> completed(nparts);
			Zeta<T> zeta(shape), eps(shape);
			std::condition_variable cv;
			std::mutex mtx;
			std::atomic<int> nfinished(0);
			/// 3. Put all partitions in a queue and process them in parallel.
			/// Each thread traverses the queue looking for partitions depedent
			/// partitions of which has been computed. When eligible partition is
			/// found, it is removed from the queue and computed and its status is
			/// updated in a separate map.
			#pragma omp parallel
			{
				std::unique_lock<std::mutex> lock(mtx);
				while (!parts.empty()) {
					typename std::vector<Partition>::iterator result;
					cv.wait(lock, [&result,&parts,&completed] () {
						result = std::find_if(
							parts.begin(),
							parts.end(),
							[&completed] (const Partition& part) {
								completed(part.ijk) = true;
								size3 ijk0 = blitz::max(0, part.ijk - 1);
								bool all_completed = blitz::all(
									completed(blitz::RectDomain<3>(ijk0, part.ijk))
								);
								completed(part.ijk) = false;
								return all_completed;
							}
						);
						return result != parts.end() || parts.empty();
					});
					if (parts.empty()) {
						break;
					}
					Partition part = *result;
					parts.erase(result);
					lock.unlock();
					eps(part.rect) = generate_white_noise(
						part.shape(),
						var_wn,
						std::ref(part.prng)
					);
					model(zeta, eps, part.rect);
					lock.lock();
					std::clog
						<< "Finished part ["
						<< ++nfinished << '/' << ntotal << ']'
						<< std::endl;
					completed(part.ijk) = true;
					cv.notify_all();
				}
			}
			return zeta;
		}

		std::vector<Partition>
		partition(
			size3 nparts,
			size3 partshape,
			size3 shape,
			const std::vector<mt_config>& prng_config
		) {
			std::vector<Partition> parts;
			const int nt = nparts(0);
			const int nx = nparts(1);
			const int ny = nparts(2);
			for (int i=0; i<nt; ++i) {
				for (int j=0; j<nx; ++j) {
					for (int k=0; k<ny; ++k) {
						const size3 ijk(i, j, k);
						const size3 lower = blitz::min(ijk * partshape, shape);
						const size3 upper = blitz::min((ijk+1) * partshape, shape) - 1;
						parts.emplace_back(
							ijk,
							blitz::RectDomain<3>(lower, upper),
							prng_config[parts.size()]
						);
					}
				}
			}
			return std::move(parts);
		}

		#endif

		size3
		get_partition_shape(size3 order, int nprngs) {
			size3 ret;
			if (blitz::product(_partition) > 0) {
				ret = _partition;
			} else {
				const size3 shape = _outgrid.size();
				const size3 guess1 = blitz::max(
					order * 2,
					size3(10, 10, 10)
				);
				const int parallelism = std::min(omp_get_max_threads(), nprngs);
				const int npar = std::max(1, 7*int(std::cbrt(parallelism)));
				const size3 guess2 = blitz::div_ceil(
					shape,
					size3(npar, npar, npar)
				);
				ret = blitz::min(guess1, guess2) + blitz::abs(guess1 - guess2) / 2;
			}
			return ret;
		}

		template <class Model>
		void
		verify(ACF<T> acf, Zeta<T> zeta, Model model) {
			switch (_vscheme) {
				case Verification_scheme::None:
					break;
				case Verification_scheme::Summary:
				case Verification_scheme::Quantile:
					show_statistics(acf, zeta, model);
					break;
				case Verification_scheme::Manual:
					write_everything_to_files(acf, zeta);
					break;
			}
		}

		template<class Model>
		void
		show_statistics(ACF<T> acf, Zeta<T> zeta, Model model) {
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
		write_everything_to_files(ACF<T> acf, Zeta<T> zeta) {
			Wave_field<T> wave_field(zeta);
			write_raw("heights_x", wave_field.heights_x());
			write_raw("heights_y", wave_field.heights_y());
			write_raw("periods", wave_field.periods());
			write_raw("lengths_x", wave_field.lengths_x());
			write_raw("lengths_y", wave_field.lengths_y());
			write_raw("elevation", zeta);
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
			    {"partition", sys::make_param(_partition)},
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

		#if !ARMA_NONE
		template<class Result>
		void
		read_parallel_mt_config(const char* filename, Result result) {
			std::ifstream in(filename);
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

		inline static clock_type::rep
		newseed() noexcept {
			#if defined(ARMA_NO_PRNG_SEED)
			return clock_type::rep(0);
			#else
			return clock_type::now().time_since_epoch().count();
			#endif
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
		size3 _partition; //< The size of partitions that are computed in parallel.

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
