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
#include "generator/basic_model.hh"
#include "output_flags.hh"
#include "params.hh"
#include "grid.hh"
#include "stats/statistics.hh"
#include "stats/distribution.hh"
#include "stats/summary.hh"
#include "stats/waves.hh"
#include "velocity/basic_solver.hh"
#include "discrete_function.hh"
#include "util.hh"
#include "bits/object_wrapper.hh"
#include "bits/write_csv.hh"

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

		Output_flags
		vscheme() const noexcept {
			return this->_generator->vscheme();
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
		write_wavy_surface(std::string filename) {
			if (this->vscheme().isset(Output_flags::Blitz)) {
				std::ofstream(filename) << _zeta;
			}
			if (this->vscheme().isset(Output_flags::CSV)) {
				bits::write_csv(filename, _zeta, _zeta.grid());
			}
		}

		void
		write_velocity_potentials(std::string filename) {
			if (this->vscheme().isset(Output_flags::Blitz)) {
				std::ofstream(filename) << _vpotentials;
			}
			if (this->vscheme().isset(Output_flags::CSV)) {
				write_4d_csv(
					filename,
					_vpotentials,
					_vpsolver->domain()
				);
			}
		}

		void
		generate_wavy_surface() {
			echo_parameters();
			this->_zeta.reference(this->_generator->generate());
			this->_generator->verify(this->_zeta);
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

		/// Read AR model parameters from an input stream.
		void
		read_parameters(std::istream& in) {
			bits::Object_wrapper<model_type> model_wrapper(
				this->_generator,
				this->_models
			);
			bits::Object_wrapper<vpsolver_type> vpsolver_wrapper(
				this->_vpsolver,
				this->_vpsolvers
			);
			sys::parameter_map params({
			    {"model", sys::make_param(model_wrapper)},
			    {"velocity_potential_solver", sys::make_param(vpsolver_wrapper)},
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
				write_key_value(std::clog, "Model", *this->_generator);
			}
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

		generator::Basic_model<T>* _generator = nullptr;
		vpsolver_type* _vpsolver = nullptr;

		Discrete_function<T,3> _zeta;
		Array4D<T> _vpotentials;

		std::unordered_map<std::string, vpsolver_ctr> _vpsolvers;
		std::unordered_map<std::string, model_ctr> _models;
	};

}

#endif // ARMA_DRIVER_HH
