#ifndef AUTOREG_DRIVER_HH
#define AUTOREG_DRIVER_HH

/// @file
/// Some abbreviations used throughout the programme.
/// AR      autoregressive
/// NIT     nonlinear intertialess transformation
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
	acf_model(acf_size.count()),
	acf_pure(acf_size.count()),
	ar_coefs(fsize.count()),
	interp_coefs(NIT_INTERPOLATION_POLYNOMIAL_ORDER)
	{
		validate_parameters();
	}

	void act() { 
	
		echo_parameters();
		compute_autoreg_coefs();
		Variance_WN<T> compute_variance(ar_coefs, acf_model);
		compute_variance.act();
		T var_wn = compute_variance.getsum();
		std::clog << "var_acf=" << var_acf(acf_model) << std::endl;
		std::clog << "var_wn=" << var_wn << std::endl;
		Wave_surface_generator<T> wavy_surface_generator(ar_coefs, fsize, var_wn,
			zsize2, zsize, zdelta);
		wavy_surface_generator.act();
		std::valarray<T>& wavy_surface = wavy_surface_generator.get_wavy_surface();
		if (!linear) {
			Skew_normal<T> cdf(skewness, kurtosis);
			transform_wavy_surface(interp_coefs, zsize, wavy_surface, cdf, nit_x0, nit_x1);
		}
		write_zeta(wavy_surface);
		std::exit(0);
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
	void read_parameters(std::istream& in) {
		std::string name;
		T size_factor = 1.2;
		while (!getline(in, name, '=').eof()) {
			if (name.size() > 0 && name[0] == '#') in.ignore(1024*1024, '\n');
			else if (name == "linear"      ) in >> linear;
			else if (name == "skewness"    ) in >> skewness;
			else if (name == "kurtosis"    ) in >> kurtosis;
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
		acf_model.resize(acf_size.count());
		acf_pure.resize(acf_size.count());
		ar_coefs.resize(fsize.count());
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
		if (sz.reduce(std::multiplies<T>()) == 0) {
			std::stringstream str;
			str << "Invalid " << var_name << ": " << sz;
			throw std::runtime_error(str.str().c_str());
		}
	}

	void echo_parameters() {
		std::clog << std::left << std::boolalpha;
		write_key_value(std::clog, "acf_size:"   , acf_size);
		write_key_value(std::clog, "zsize:"      , zsize);
		write_key_value(std::clog, "zsize2:"     , zsize2);
		write_key_value(std::clog, "zdelta:"     , zdelta);
		write_key_value(std::clog, "linear:"     , linear);
		write_key_value(std::clog, "skewness:"   , skewness);
		write_key_value(std::clog, "kurtosis:"   , kurtosis);
		write_key_value(std::clog, "size_factor:", size_factor());
	}

	template<class V>
	std::ostream&
	write_key_value(std::ostream& out, const char* key, V value) {
		return out << std::setw(20) << key << value << std::endl;
	}

	template<class CDF>
	void write_wave_distribution(CDF cdf) {
		std::ofstream out("skew_normal");
		int count = 100;
		T x0 = -5;
		T x1 = 5;
		T dx = (x1-x0)/count;
		for (int i=0; i<100; ++i) {
			T x = x0 + i*dx;
			out << x << ' ' << cdf(x) << std::endl;
		}
	}

	void compute_autoreg_coefs() {
		acf_pure = acf_model;
		if (!linear) {
			Skew_normal<T> cdf(skewness, kurtosis);
			//Skew_normal_2<T> cdf(COEF);
			//Weibull<T> cdf(3, 1);
			T breadth = NIT_SIGMA_COUNT*sqrt(var_acf(acf_model));
			nit_x0 = -breadth;
			nit_x1 =  breadth;
#ifdef DEBUG
			write_wave_distribution(cdf);
#endif
			interpolation_coefs<T>(nit_x0, nit_x1, NIT_INTERPOLATION_NODES, interp_coefs, cdf);
			transform_acf<T>(interp_coefs, NIT_MAX_COEFS, acf_model);
		}
		Autoreg_coefs<T> compute_coefs(acf_model, acf_size, ar_coefs);
		compute_coefs.act();
		{ std::ofstream out("ar_coefs"); out << ar_coefs; }
	}

	void write_zeta(const std::valarray<T>& zeta) {
		std::ofstream out("zeta");
		out << Domain<T, 3>(zdelta, zsize) << std::endl;
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

	/// Parameters used in nonlinear inertialess transfromation.
	bool linear = false;
	T skewness = 0;
	T kurtosis = 0;
	T nit_x0 = 0;
	T nit_x1 = 0;

	/// ACF transformed by NIT (equals to @acf_pure when linear=true).
	std::valarray<T> acf_model;

	/// Initial ACF which is not transformed by NIT (equals to @acf_model when linear=true).
	std::valarray<T> acf_pure;

	/// AR model coefficients.
	std::valarray<T> ar_coefs;

	/// NIT interpolations coefficients.
	std::valarray<T> interp_coefs;
	
	/// ACF parameters
	/// @see approx_acf
	T alpha = 0.05;
	T beta = 0.8;
	T gamm = 1.0;

	/// Some constants for nonlinear inertialess transfromation.
	static const std::size_t NIT_INTERPOLATION_POLYNOMIAL_ORDER = 12;
	static const std::size_t NIT_INTERPOLATION_NODES            = 100;
	static const std::size_t NIT_MAX_COEFS                      = 10;
	static constexpr T       NIT_SIGMA_COUNT                    = 3.0;
};

}

#endif // AUTOREG_DRIVER_HH
