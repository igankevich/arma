#ifndef AUTOREG_DRIVER_HH
#define AUTOREG_DRIVER_HH

#include <iostream>  // for operator<<, basic_ostream, clog
#include <iomanip>   // for operator<<, setw
#include <fstream>   // for ofstream
#include <stdexcept> // for runtime_error
#include <string>    // for operator==, basic_string, string, getline

#include "types.hh"   // for size3, Vector, Zeta, ACF, AR_coefs
#include "autoreg.hh" // for mean, variance, ACF_variance, approx_acf, comp...
#include "params.hh"
#include "grid.hh"

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

namespace autoreg {

	/// Class that reads parameters from the input files,
	/// calls all subroutines, and prints the result.
	template <class T>
	struct Autoreg_model {

		Autoreg_model()
		    : _outgrid{{768, 24, 24}},
		      _acfgrid{{10, 10, 10}, {T(2.5), T(5), T(5)}},
		      _arorder(_acfgrid.size()) {}

		void
		act() {
			echo_parameters();
			ACF<T> acf_model =
			    standing_wave_ACF<T>(_acfgrid.delta(), _acfgrid.size());
			{
				std::ofstream out("acf");
				out << acf_model;
			}
			AR_coefs<T> ar_coefs =
			    compute_AR_coefs(acf_model, _arorder, _doleastsquares);
			T var_wn = white_noise_variance(ar_coefs, acf_model);
			std::clog << "ACF variance = " << ACF_variance(acf_model)
			          << std::endl;
			std::clog << "WN variance = " << var_wn << std::endl;
			Zeta<T> zeta = generate_white_noise(_outgrid.size(), var_wn);
			std::clog << "mean(eps) = " << mean(zeta) << std::endl;
			std::clog << "variance(eps) = " << variance(zeta) << std::endl;
			generate_zeta(ar_coefs, zeta);
			std::clog << "mean(zeta) = " << mean(zeta) << std::endl;
			std::clog << "variance(zeta) = " << variance(zeta) << std::endl;
			write_zeta(zeta);
		}

		/// Read AR model parameters from an input stream, generate default ACF
		/// and
		/// validate all the parameters.
		template <class V>
		friend std::istream& operator>>(std::istream& in, Autoreg_model<V>& m) {
			m.read_parameters(in);
			m.validate_parameters();
			return in;
		}

	private:
		/// Read AR model parameters from an input stream.
		void
		read_parameters(std::istream& in) {
			sys::parameter_map params({
			    {"out_grid", sys::make_param(_outgrid)},
			    {"acf_grid", sys::make_param(_acfgrid)},
			    {"ar_order", sys::make_param(_arorder)},
			    {"least_squares", sys::make_param(_doleastsquares)},
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
			check_non_zero(_arorder, "AR order");
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
			std::clog << std::left;
			write_key_value(std::clog, "ACF grid size", _acfgrid.size());
			write_key_value(std::clog, "ACF grid patch size",
			                _acfgrid.patch_size());
			write_key_value(std::clog, "Output grid size", _outgrid.size());
			write_key_value(std::clog, "Output grid patch size",
			                _outgrid.patch_size());
			write_key_value(std::clog, "AR order", _arorder);
			write_key_value(std::clog, "Do least squares", _doleastsquares);
		}

		template <class V>
		std::ostream&
		write_key_value(std::ostream& out, const char* key, V value) {
			return out << std::setw(30) << key << " = " << value << std::endl;
		}

		void
		write_zeta(const Zeta<T>& zeta) {
			std::ofstream out("zeta");
			out << zeta;
		}

		Grid<T, 3> _outgrid; //< Wavy surface grid.
		Grid<T, 3> _acfgrid; //< ACF grid.
		size3 _arorder;      //< AR model order (no. of coefficients).
		bool _doleastsquares = false;
	};
}

#endif // AUTOREG_DRIVER_HH
