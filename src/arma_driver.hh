#ifndef ARMA_DRIVER_HH
#define ARMA_DRIVER_HH

#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <functional>
#include <type_traits>
#include <iterator>

#include "config.hh"
#include "types.hh"
#include "arma.hh"
#include "generator/ar_model.hh"
#include "generator/ma_model.hh"
#include "generator/arma_model.hh"
#include "generator/plain_wave.hh"
#include "generator/lh_model.hh"
#include "simulation_model.hh"
#include "verification_scheme.hh"
#include "acf.hh"
#include "params.hh"
#include "grid.hh"
#include "stats/statistics.hh"
#include "stats/distribution.hh"
#include "stats/summary.hh"
#include "stats/waves.hh"
#include "output_format.hh"
#include "velocity/basic_solver.hh"
#include "discrete_function.hh"
#include "util.hh"
#include "nonlinear/nit_transform.hh"

/** \mainpage
Some abbreviations used throughout the programme.
| Abbreviation      | Meaning                 |
|:------------------|:------------------------|
| AR                | autoregressive          |
| ACF               | auto-covariate function |
| zeta              | ocean wavy surface      |
| phi               | AR model coefficients   |
| YW                | Yule-Walker             |
| WN                | white noise             |
| var               | variance                |
| MT                | Mersenne Twister        |
*/

namespace arma {

	namespace bits {

		/// Helper class that initialises velocity potential solver by name.
		template <class T>
		class Object_wrapper {
			typedef T solver_type;
			typedef std::string key_type;
			typedef std::function<solver_type*()> value_type;
			typedef std::unordered_map<std::string, value_type> map_type;
			solver_type*& _solver;
			const map_type& _constructors;

		public:
			explicit
			Object_wrapper(solver_type*& solver, const map_type& constructors):
			_solver(solver),
			_constructors(constructors)
			{}

			friend std::istream&
			operator>>(std::istream& in, Object_wrapper& rhs) {
				std::string name;
				in >> std::ws >> name;
				auto result = rhs._constructors.find(name);
				if (result == rhs._constructors.end()) {
					in.setstate(std::ios::failbit);
					std::cerr << "Invalid object: " << name << std::endl;
					throw std::runtime_error("bad object");
				} else {
					rhs._solver = result->second();
					in >> *rhs._solver;
				}
				return in;
			}
		};

		template <class Tr>
		class Transform_wrapper {
			Tr& _transform;
			bool& _linear;
		public:
			explicit
			Transform_wrapper(Tr& tr, bool& linear):
			_transform(tr),
			_linear(linear)
			{}
			friend std::istream&
			operator>>(std::istream& in, Transform_wrapper& rhs) {
				std::string name;
				in >> std::ws >> name;
				if (name == "nit") {
					in >> rhs._transform;
					rhs._linear = false;
				} else if (name == "none") {
					rhs._linear = true;
				} else {
					in.setstate(std::ios::failbit);
					std::cerr << "Invalid transform: " << name << std::endl;
					throw std::runtime_error("bad transform");
				}
				return in;
			}
		};

	}


	/**
	\brief Control class that generates wavy surface and computes
	velocity potential field.

	Class that reads parameters from the input files,
	calls all subroutines, and prints the result.
	*/
	template <class T>
	struct ARMA_driver {

		typedef velocity::Velocity_potential_solver<T>
			vpsolver_type;
		typedef std::function<vpsolver_type*()> vpsolver_ctr;
		typedef generator::Basic_model<T> model_type;
		typedef std::function<model_type*()> model_ctr;
		typedef nonlinear::NIT_transform<T> transform_type;

		ARMA_driver() = default;

		virtual ~ARMA_driver() {
			delete _generator;
			delete _vpsolver;
		}

		const Array3D<T>&
		wavy_surface() const noexcept {
			return _zeta;
		}

		const Grid<T,3>&
		wavy_surface_grid() const noexcept {
			return this->_generator->grid();
		}

		const Array4D<T>&
		velocity_potentials() const noexcept {
			return _vpotentials;
		}

		const vpsolver_type*
		velocity_potential_solver() const noexcept {
			return _vpsolver;
		}

		vpsolver_type*
		velocity_potential_solver() noexcept {
			return _vpsolver;
		}

		Verification_scheme
		vscheme() const noexcept {
			return _vscheme;
		}

		Grid<T,3>
		velocity_potential_grid() const {
			const Grid<T,3> outgrid = this->_generator->grid();
			const int nz = _vpsolver->domain().num_points(1);
			const int nx = outgrid.num_points(1);
			const int ny = outgrid.num_points(2);
			return Grid<T,3>(
				{nz, nx, ny},
				{
					_vpsolver->domain().length(1),
					outgrid.length(1),
					outgrid.length(2)
				}
			);
		}

		void
		write_wavy_surface(std::string filename, Output_format fmt) {
			switch (fmt) {
				case Output_format::Blitz:
					std::ofstream(filename) << _zeta;
					break;
				case Output_format::CSV:
					write_csv(filename, _zeta);
					break;
				default:
					throw std::runtime_error("bad format");
			}
		}

		void
		write_velocity_potentials(std::string filename, Output_format fmt) {
			switch (fmt) {
				case Output_format::Blitz:
					std::ofstream(filename) << _vpotentials;
					break;
				case Output_format::CSV:
					write_4d_csv(
						filename,
						_vpotentials,
						_vpsolver->domain()
					);
					break;
				default:
					throw std::runtime_error("bad format");
			}
		}

		void
		generate_wavy_surface() {
			echo_parameters();
			const Grid<T,3> outgrid = this->wavy_surface_grid();
			if (_model == Simulation_model::Autoregressive) {
				generate_wavy_surface(_armodel);
			} else if (_model == Simulation_model::Moving_average) {
				generate_wavy_surface(_mamodel);
			} else if (_model == Simulation_model::ARMA) {
				generate_wavy_surface(_armamodel);
			} else if (_model == Simulation_model::Longuet_Higgins) {
				this->_zeta.reference(this->_generator->generate());
			} else if (_model == Simulation_model::Plain_wave_model) {
				this->_zeta.reference(this->_generator->generate());
			}
		}

		void
		compute_velocity_potentials() {
			_zeta.setgrid(this->wavy_surface_grid());
			this->_vpotentials.reference(_vpsolver->operator()(_zeta));
		}

		template<class Type>
		void
		register_velocity_potential_solver(std::string key) {
			_vpsolvers.emplace(key, [] () { return new Type; });
		}

		template<class Type>
		void
		register_model(const std::string& key) {
			this->_models.emplace(key, [] () { return new Type; });
		}

		/**
		Read and validate driver parameters from an input stream.
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
			Discrete_function<T,3> acf = model.acf();
			if (!_linear) {
				auto copy = acf.copy();
				_nittransform.transform_ACF(copy);
			}
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
			Array3D<T> zeta = model.generate();
//			Array3D<T> zeta = do_generate_wavy_surface(model, var_wn);
			this->_zeta.reference(zeta);
			if (!_linear) {
				_nittransform.transform_realisation(acf, _zeta);
			}
			if (std::is_same<Model,generator::AR_model<T>>::value) {
				/// Estimate mean/variance with ramp-up region removed.
				blitz::RectDomain<3> subdomain(model.order(), zeta.shape() - 1);
				Shape3D zeta_size = subdomain.ubound() - subdomain.lbound();
				std::clog << "Zeta size = " << zeta_size << std::endl;
				verify(model.acf(), zeta(subdomain), model);
			} else {
				verify(model.acf(), zeta, model);
			}
		}

		template <class Model>
		void
		verify(Array3D<T> acf, Array3D<T> zeta, Model model) {
			switch (_vscheme) {
				case Verification_scheme::No_verification:
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
		show_statistics(Array3D<T> acf, Array3D<T> zeta, Model model) {
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
			if (_vscheme == Verification_scheme::Quantile) {
				std::for_each(
					stats.begin(),
					stats.end(),
					std::mem_fn(&Summary<T>::write_quantile_graph)
				);
			}
		}

		void
		write_everything_to_files(Array3D<T> acf, Array3D<T> zeta) {
			stats::Wave_field<T> wave_field(zeta);
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
			bits::Object_wrapper<model_type> model_wrapper(
				this->_generator,
				this->_models
			);
			bits::Object_wrapper<vpsolver_type> vpsolver_wrapper(
				_vpsolver,
				_vpsolvers
			);
			bits::Transform_wrapper<transform_type> trans_wrapper(
				_nittransform,
				_linear
			);
			sys::parameter_map params({
			    {"ar_model", sys::make_param(_armodel)},
			    {"ma_model", sys::make_param(_mamodel)},
			    {"arma_model", sys::make_param(_armamodel)},
			    {"plain_wave", sys::make_param(_plainwavemodel)},
			    {"lh_model", sys::make_param(_lhmodel)},
			    {"model", sys::make_param(model_wrapper)},
			    {"verification", sys::make_param(_vscheme)},
			    {"velocity_potential_solver", sys::make_param(vpsolver_wrapper)},
			    {"transform", sys::make_param(trans_wrapper)},
			});
			in >> params;
			if (!_vpsolver) {
				std::cerr
					<< "Bad \"velocity_potential_solver\": null"
					<< std::endl;
				throw std::runtime_error("bad solver");
			}
			if (!_generator) {
				std::cerr
					<< "Bad \"generator\": null"
					<< std::endl;
				throw std::runtime_error("bad generator");
			}
		}

		void
		echo_parameters() {
			if (_generator) {
				write_key_value(
					std::clog,
					"Output grid size",
					this->_generator->grid().size()
				);
				write_key_value(
					std::clog,
					"Output grid patch size",
					this->_generator->grid().patch_size()
				);
				write_key_value(std::clog, "Model", this->_generator);
			}
			write_key_value(std::clog, "Verification scheme", _vscheme);
			/*
			switch (_model) {
				case Simulation_model::Autoregressive:
					write_key_value(std::clog, "AR model", _armodel);
					break;
				case Simulation_model::Moving_average:
					write_key_value(std::clog, "MA model", _mamodel);
					break;
				case Simulation_model::ARMA:
					write_key_value(
						std::clog,
						"ARMA model",
						"<not implemented>"
					);
					break;
				case Simulation_model::Plain_wave_model:
					write_key_value(
						std::clog,
						"Plain wave model",
						_plainwavemodel
					);
					break;
				case Simulation_model::Longuet_Higgins:
					write_key_value(
						std::clog,
						"Longuet-Higgins model",
						_lhmodel
					);
					break;
			}
			*/
			if (_vpsolver) {
				write_key_value(
					std::clog,
					"Velocity potential solver name",
					typeid(*_vpsolver).name()
				);
				write_key_value(
					std::clog,
					"Velocity potential solver",
					*_vpsolver
				);
			}
			if (_linear) {
				write_key_value(std::clog, "NIT transform", "none");
			} else {
				write_key_value(std::clog, "NIT transform", _nittransform);
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
			std::string filename,
			const blitz::Array<X, 3>& data,
			const char separator=','
		) {
			const Grid<T,3> outgrid = this->_generator->grid();
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
						const T x = outgrid(j, 1);
						const T y = outgrid(k, 2);
						out << i << separator
							<< x << separator
							<< y << separator
							<< data(i, j, k)
							<< '\n';
					}
				}
			}
		}

		template<class X>
		void
		write_4d_csv(
			const std::string& filename,
			const blitz::Array<X, 4>& data,
			const Domain2<T>& domain,
			const char separator=','
		) {
			const Grid<T,3> outgrid = this->_generator->grid();
			std::ofstream out(filename);
			out << 't' << separator
			   << 'z' << separator
			   << 'x' << separator
			   << 'y' << separator
			   << "phi" << '\n';
			const int nt = data.extent(0);
			const int nz = data.extent(1);
			const int nx = data.extent(2);
			const int ny = data.extent(3);
			for (int i=0; i<nt; ++i) {
				for (int j=0; j<nz; ++j) {
					const Vec2D<T> p = domain({i,j});
					for (int k=0; k<nx; ++k) {
						for (int l=0; l<ny; ++l) {
							const T x = outgrid(k, 1);
							const T y = outgrid(l, 2);
							out << p(0) << separator
								<< p(1) << separator
								<< x << separator
								<< y << separator
								<< data(i, j, k, l)
								<< '\n';
						}
					}
				}
			}
		}

		Simulation_model _model = Simulation_model::Autoregressive;
		Verification_scheme _vscheme = Verification_scheme::Summary;

		generator::Basic_model<T>* _generator = nullptr;
		generator::AR_model<T> _armodel;
		generator::MA_model<T> _mamodel;
		generator::ARMA_model<T> _armamodel;
		generator::Plain_wave_model<T> _plainwavemodel;
		generator::Longuet_Higgins_model<T> _lhmodel;

		vpsolver_type* _vpsolver = nullptr;

		Discrete_function<T,3> _zeta;
		Array4D<T> _vpotentials;
		nonlinear::NIT_transform<T> _nittransform;
		bool _linear = true;

		std::unordered_map<std::string, vpsolver_ctr> _vpsolvers;
		std::unordered_map<std::string, model_ctr> _models;
	};

}

#endif // ARMA_DRIVER_HH
