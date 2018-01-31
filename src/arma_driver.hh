#ifndef ARMA_DRIVER_HH
#define ARMA_DRIVER_HH

#include <functional>
#include <istream>
#include <ostream>
#include <string>
#include <unordered_map>

#include "config.hh"
#include "types.hh"
#include "arma.hh"
#include "generator/basic_model.hh"
#include "output_flags.hh"
#include "grid.hh"
#include "velocity/basic_solver.hh"
#include "discrete_function.hh"
#include "util.hh"

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

	public:
		typedef velocity::Velocity_potential_solver<T>
			vpsolver_type;
		typedef std::function<vpsolver_type*()> vpsolver_ctr;
		typedef generator::Basic_model<T> model_type;
		typedef std::function<model_type*()> model_ctr;

	protected:
		model_type* _model = nullptr;
		vpsolver_type* _solver = nullptr;
		Discrete_function<T,3> _zeta;
		Array4D<T> _vpotentials;
		std::unordered_map<std::string, vpsolver_ctr> _solvers;
		std::unordered_map<std::string, model_ctr> _models;
		std::string _solvername;

	public:
		ARMA_driver() = default;

		inline virtual
		~ARMA_driver() {
			#if !ARMA_BSCHEDULER
			delete _model;
			#endif
			delete _solver;
		}

		void
		open(const std::string& filename);

		inline const Array3D<T>&
		wavy_surface() const noexcept {
			return _zeta;
		}

		inline const Grid<T,3>&
		wavy_surface_grid() const noexcept {
			return this->_model->grid();
		}

		inline const Array4D<T>&
		velocity_potentials() const noexcept {
			return _vpotentials;
		}

		inline const vpsolver_type*
		velocity_potential_solver() const noexcept {
			return _solver;
		}

		inline vpsolver_type*
		velocity_potential_solver() noexcept {
			return _solver;
		}

		inline Output_flags
		oflags() const noexcept {
			return this->_model->oflags();
		}

		Grid<T,3>
		velocity_potential_grid() const;

		void
		write_wavy_surface();

		void
		write_velocity_potentials();

		void
		write_all();

		void
		generate_wavy_surface();

		void
		compute_velocity_potentials();

		template<class Type>
		void
		register_solver(std::string key) {
			this->_solvers.emplace(key, [] () { return new Type; });
		}

		template<class Type>
		void
		register_model(const std::string& key) {
			this->_models.emplace(key, [] () { return new Type; });
		}

		virtual void
		read(std::istream& in);

		/**
		Read and validate driver parameters from an input stream.
		*/
		template <class X>
		friend std::istream&
		operator>>(std::istream& in, ARMA_driver<X>& m);

	protected:
		void
		echo_parameters();

	};

	template <class T>
	std::istream&
	operator>>(std::istream& in, ARMA_driver<T>& m);

}

#endif // ARMA_DRIVER_HH
