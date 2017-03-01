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
#include "models/plain_wave.hh"
#include "simulation_model.hh"
#include "verification_scheme.hh" // for Verification_scheme
#include "acf.hh" // for standing_wave_ACF, propagating_wave_ACF
#include "params.hh"
#include "grid.hh"
#include "statistics.hh"
#include "distribution.hh"
#include "parallel_mt.hh"
#include "errors.hh"
#include "velocity/velocity_potential_field.hh"
#include "velocity/linear_velocity_potential_field.hh"

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

	template<class T>
	std::istream&
	operator>>(std::istream& in, Velocity_potential_field<T>*& rhs) {
		std::string name;
		in >> std::ws >> name;
		if (name == "linear") {
			auto tmp = new Linear_velocity_potential_field<T>;
			in >> *tmp;
			rhs = tmp;
		} else {
			in.setstate(std::ios::failbit);
			std::clog << "Invalid velocity field: " << name << std::endl;
			throw std::runtime_error("bad velocity field");
		}
		return in;
	}

	/// Class that reads parameters from the input files,
	/// calls all subroutines, and prints the result.
	template <class T>
	struct ARMA_driver {

		typedef std::chrono::high_resolution_clock clock_type;

		ARMA_driver():
		_outgrid{{768, 24, 24}},
		_partition(0,0,0)
		{}

		~ARMA_driver() {
			delete _velocityfield;
		}

		void
		act() {
			echo_parameters();
			if (_model == Simulation_model::Autoregressive) {
				generate_wavy_surface(_armodel);
			} else if (_model == Simulation_model::Moving_average) {
				generate_wavy_surface(_mamodel);
			} else if (_model == Simulation_model::ARMA) {
				generate_wavy_surface(_armamodel);
			} else if (_model == Simulation_model::Plain_wave) {
				Zeta<T> zeta(_outgrid.size());
			   	_plainwavemodel(zeta);
				write_zeta(zeta);
			}
		}

		/**
		Read AR model parameters from an input stream, generate default ACF
		and validate all the parameters.
		*/
		template <class V>
		friend std::istream&
		operator>>(std::istream& in, ARMA_driver<V>& m) {
			m.read_parameters(in);
			return in;
		}

	private:
		template <class Model>
		void
		generate_wavy_surface(Model& model) {
			ACF<T> acf = model.acf();
			std::clog << "ACF variance = " << ACF_variance(acf) << std::endl;
			if (_vscheme == Verification_scheme::Manual) {
				write_csv("acf.csv", acf);
			}
			{
				std::ofstream out("acf");
				out << acf;
			}
			model.determine_coefficients();
			model.validate();
			T var_wn = model.white_noise_variance();
			std::clog << "WN variance = " << var_wn << std::endl;
			Zeta<T> zeta = do_generate_wavy_surface(model, var_wn);
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

		template <class Model>
		Zeta<T>
		do_generate_wavy_surface(Model& model, T var_wn) {
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

		template <class Model>
		Zeta<T>
		do_generate_wavy_surface(Model& model, T var_wn) {
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
			    {"out_grid", sys::make_param(_outgrid, validate_grid<T,3>)},
			    {"ar_model", sys::make_param(_armodel)},
			    {"ma_model", sys::make_param(_mamodel)},
			    {"arma_model", sys::make_param(_armamodel)},
			    {"plain_wave", sys::make_param(_plainwavemodel)},
			    {"model", sys::make_param(_model)},
			    {"verification", sys::make_param(_vscheme)},
			    {"partition", sys::make_param(_partition)},
			    {"velocity_field", sys::make_param(
					static_cast<Velocity_potential_field<T>*&>(_velocityfield)
			    )},
			});
			in >> params;
		}

		void
		echo_parameters() {
			write_key_value(std::clog, "Output grid size", _outgrid.size());
			write_key_value(std::clog, "Output grid patch size",
			                _outgrid.patch_size());
			write_key_value(std::clog, "Model", _model);
			write_key_value(std::clog, "Verification scheme", _vscheme);
			switch (_model) {
				case Simulation_model::Autoregressive:
					write_key_value(std::clog, "AR model", _armodel);
					break;
				case Simulation_model::Moving_average:
					write_key_value(std::clog, "MA model", _mamodel);
					break;
				case Simulation_model::ARMA:
					write_key_value(std::clog, "ARMA model", "<not implemented>");
					break;
				case Simulation_model::Plain_wave:
					write_key_value(std::clog, "Plain wave model", _plainwavemodel);
					break;
				case Simulation_model::Longuet_Higgins:
					write_key_value(
						std::clog,
						"Longuet-Higgins model",
						"<not implemented>"
					);
					break;
			}
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

		inline static clock_type::rep
		newseed() noexcept {
			#if defined(ARMA_NO_PRNG_SEED)
			return clock_type::rep(0);
			#else
			return clock_type::now().time_since_epoch().count();
			#endif
		}

		Grid<T, 3> _outgrid; //< Wavy surface grid.

		Simulation_model _model = Simulation_model::Autoregressive;
		Verification_scheme _vscheme = Verification_scheme::Summary;
		size3 _partition; //< The size of partitions that are computed in parallel.

		Autoregressive_model<T> _armodel;
		Moving_average_model<T> _mamodel;
		ARMA_model<T> _armamodel;
		Plain_wave_model<T> _plainwavemodel;

		Velocity_potential_field<T>* _velocityfield = nullptr;
	};

}

#endif // ARMA_DRIVER_HH
