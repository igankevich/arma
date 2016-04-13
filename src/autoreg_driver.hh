#ifndef AUTOREG_DRIVER_HH
#define AUTOREG_DRIVER_HH

#include <iostream>
#include <iomanip>
#include <fstream>

#include "valarray_ext.hh"
#include "vector_n.hh"

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

/// Class that reads paramters from the input files,
/// calls all subroutines, and prints the result.
template<class T>
class Autoreg_model {
public:

	Autoreg_model():
	zsize(768, 24, 24),
	zdelta(1, 1, 1),
	acf_size(10, 10, 10),
	acf_delta(zdelta),
	fsize(acf_size),
	zsize2(zsize),
	acf_model(blitz::product(acf_size)),
	ar_coefs(blitz::product(fsize))
	{}

	void act() {
		echo_parameters();
		compute_autoreg_coefs();
		T var_wn = white_noise_variance(ar_coefs, acf_model);
		std::clog << "ACF variance = " << ACF_variance(acf_model) << std::endl;
		std::clog << "WN variance = " << var_wn << std::endl;
		std::valarray<T> zeta(blitz::product(zsize));
		std::valarray<T> zeta2(blitz::product(zsize2));
		generate_white_noise(zeta2, var_wn);
		generate_zeta(ar_coefs, fsize, zsize2, zeta2);
		trim_zeta(zeta2, zsize2, zsize, zeta);
		write_zeta(zeta);
	}

	/// Read AR model parameters from an input stream, generate default ACF and
	/// validate all the parameters.
	template<class V>
	friend std::istream&
	operator>>(std::istream& in, Autoreg_model<V>& m) {

		m.read_parameters(in);

		// generate ACF
		approx_acf<T>(m.alpha, m.beta, m.gamm, m.acf_delta, m.acf_size, m.acf_model);

		m.validate_parameters();

		return in;
	}

private:

	T size_factor() const { return T(zsize2[0]) / T(zsize[0]); }

	/// Read AR model parameters from an input stream.
	void
	read_parameters(std::istream& in) {
		std::string name;
		T size_factor = 1.2;
		while (!getline(in, name, '=').eof()) {
			if (name.size() > 0 && name[0] == '#') in.ignore(1024*1024, '\n');
			else if (name == "zsize"       ) in >> zsize;
			else if (name == "zdelta"      ) in >> zdelta;
			else if (name == "acf_size"    ) in >> acf_size;
			else if (name == "size_factor" ) in >> size_factor;
			else if (name == "alpha"       ) in >> alpha;
			else if (name == "beta"        ) in >> beta;
			else if (name == "gamma"       ) in >> gamm;
			else {
				in.ignore(1024*1024, '\n');
				std::stringstream str;
				str << "Unknown parameter: " << name << '.';
				throw std::runtime_error(str.str().c_str());
			}
			in >> std::ws;
		}

		if (size_factor < T(1)) {
			std::stringstream str;
			str << "Invalid size factor: " << size_factor;
			throw std::runtime_error(str.str().c_str());
		}

		zsize2 = size3(Vector<T,3>(zsize)*size_factor);
		acf_delta = zdelta;
		fsize = acf_size;
		acf_model.resize(blitz::product(acf_size));
		ar_coefs.resize(blitz::product(fsize));
	}

	/// Check for common input/logical errors and numerical implementation constraints.
	void validate_parameters() {
		check_non_zero(zsize, "zsize");
		check_non_zero(zdelta, "zdelta");
		check_non_zero(acf_size, "acf_size");
		for (int i=0; i<3; ++i) {
			if (zsize2[i] < zsize[i]) {
				throw std::runtime_error("size_factor < 1, zsize2 < zsize");
			}
		}
		int part_sz = zsize[0];
		int fsize_t = fsize[0];
		if (fsize_t > part_sz) {
			std::stringstream tmp;
			tmp << "fsize[0] > zsize[0], should be 0 < fsize[0] < zsize[0]\n";
			tmp << "fsize[0]  = " << fsize_t << '\n';
			tmp << "zsize[0] = " << part_sz << '\n';
			throw std::runtime_error(tmp.str());
		}
	}

	/// Check that all components of vector @sz are non-zero,
	/// i.e. it is valid size specification.
	template<class V>
	void check_non_zero(const Vector<V, 3>& sz, const char* var_name) {
		if (blitz::product(sz) == 0) {
			std::stringstream str;
			str << "Invalid " << var_name << ": " << sz;
			throw std::runtime_error(str.str().c_str());
		}
	}

	void
	echo_parameters() {
		std::clog << std::left;
		write_key_value(std::clog, "acf_size:"   , acf_size);
		write_key_value(std::clog, "zsize:"      , zsize);
		write_key_value(std::clog, "zsize2:"     , zsize2);
		write_key_value(std::clog, "zdelta:"     , zdelta);
		write_key_value(std::clog, "size_factor:", size_factor());
	}

	template<class V>
	std::ostream&
	write_key_value(std::ostream& out, const char* key, V value) {
		return out << std::setw(20) << key << value << std::endl;
	}

	void compute_autoreg_coefs() {
		Autoreg_coefs<T> compute_coefs(acf_model, acf_size, ar_coefs);
		compute_coefs.act();
		{ std::ofstream out("ar_coefs"); out << ar_coefs; }
	}

	void write_zeta(const std::valarray<T>& zeta) {
		std::ofstream out("zeta");
		const Vector<T,3> lower(0, 0, 0);
		const Vector<T,3> upper = zdelta * (zsize - Vector<T,3>(1, 1, 1));
		out << lower << '\n';
		out << upper << '\n';
		out << zsize << '\n';
		const Index<3> idz(zsize);
		const int t0 = 0;
		const int t1 = zsize[0];
		const int x1 = zsize[1];
		const int y1 = zsize[2];
	    for (int t=t0; t<t1; t++) {
	        for (int x=0; x<x1; x++) {
	            for (int y=0; y<y1; y++) {
					out << zeta[idz(t, x, y)] << ' ';
				}
				out << '\n';
			}
		}
	}

	/// Wavy surface size.
	size3 zsize;

	/// Wavy surface grid granularity.
	Vector<T, 3> zdelta;

	/// Auto-covariate function size.
	size3 acf_size;

	/// Auto-covariate function grid granularity.
	Vector<T, 3> acf_delta;

	/// Size of the array of AR coefficients.
	size3 fsize;

	/// Size of enlarged wavy surface. Equals to @zsize multiplied
	/// by size_factor read from input file.
	size3 zsize2;

	/// ACF transformed by NIT
	std::valarray<T> acf_model;

	/// AR model coefficients.
	std::valarray<T> ar_coefs;

	/// ACF parameters
	/// @see approx_acf
	T alpha = 0.05;
	T beta = 0.8;
	T gamm = 1.0;

};

}

#endif // AUTOREG_DRIVER_HH
