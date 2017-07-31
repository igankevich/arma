#include "plain_wave_model.hh"
#include "validators.hh"
#include "params.hh"
#include "domain.hh"
#include "profile.hh"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>

namespace {

	template<class T>
	class Array_wrapper {

		arma::Array1D<T>& _arr;

	public:

		explicit
		Array_wrapper(arma::Array1D<T>& arr):
		_arr(arr)
		{}

		explicit
		Array_wrapper(const arma::Array1D<T>& arr):
		_arr(const_cast<arma::Array1D<T>&>(arr))
		{}

		friend std::istream&
		operator>>(std::istream& in, Array_wrapper& rhs) {
			char ch = 0;
			if (in >> std::ws >> ch && ch != '[') {
				in.setstate(std::ios::failbit);
			}
			std::vector<T> values;
			while (in >> std::ws) {
				ch = in.get();
				if (ch == ']') {
					break;
				}
				in.putback(ch);
				values.emplace_back();
				in >> values.back();
			}
			rhs._arr.resize(values.size());
			std::copy(values.begin(), values.end(), rhs._arr.begin());
			return in;
		}

		friend std::ostream&
		operator<<(std::ostream& out, const Array_wrapper& rhs) {
			out << '[';
			const int n = rhs._arr.extent(0);
			if (n > 0) {
				out << rhs._arr(0);
				for (int i=1; i<n; ++i) {
					out << ' ' << rhs._arr(i);
				}
			}
			out << ']';
			return out;
		}
	};

}

std::istream&
arma::generator::bits::operator>>(std::istream& in, Function& rhs) {
	std::string name;
	in >> std::ws >> name;
	if (name == "sin") {
		rhs = Function::Sine;
	} else if (name == "cos") {
		rhs = Function::Cosine;
	} else {
		in.setstate(std::ios::failbit);
		std::cerr << "Invalid plain wave function: " << name << std::endl;
		throw std::runtime_error("bad function");
	}
	return in;
}

const char*
arma::generator::bits::to_string(Function rhs) {
	switch (rhs) {
		case Function::Sine: return "sin";
		case Function::Cosine: return "cos";
		default: return "UNKNOWN";
	}
}

template <class T>
void
arma::generator::Plain_wave_model<T>::validate() const {
	for (const wave_type& w : this->_waves) {
		validate_positive(w(0), "amplitudes");
		validate_positive(w(1), "wavenumbers_x");
		validate_positive(w(2), "wavenumbers_y");
		for (int i=0; i<w.length(); ++i) {
			validate_finite(w(i), "waves");
		}
	}
}

template <class T>
arma::Array3D<T>
arma::generator::Plain_wave_model<T>::generate() {
	Array3D<T> zeta;
	ARMA_PROFILE_BLOCK("generate_surface",
		zeta.reference(this->do_generate());
	);
	return zeta;
}

template <class T>
arma::Array3D<T>
arma::generator::Plain_wave_model<T>::do_generate() {
	Array3D<T> zeta(this->grid().num_points());
	using constants::_2pi;
	using std::sin;
	const T shift = get_shift();
	const int t1 = zeta.extent(0);
	const int j1 = zeta.extent(1);
	const int k1 = zeta.extent(2);
	const grid_type& grid = this->grid();
	const int nwaves = this->num_waves();
	#if ARMA_OPENMP
	#pragma omp parallel for collapse(3)
	#endif
	for (int t = 0; t < t1; t++) {
		for (int j = 0; j < j1; j++) {
			for (int k = 0; k < k1; k++) {
				const T x = grid(j, 1);
				const T y = grid(k, 2);
				T sum  = 0;
				for (int i=0; i<nwaves; ++i) {
					sum += amplitude(i) * sin(
						_2pi<T>*wavenum_x(i)*x + _2pi<T>*wavenum_y(i)*y
						- velocity(i)*t + phase(i) + shift
					);
				}
				zeta(t, j, k) = sum;
			}
		}
	}
	return zeta;
}


template <class T>
void
arma::generator::Plain_wave_model<T>::read(std::istream& in) {
	Array_wrapper<wave_type> waves_wrappper(this->_waves);
	sys::parameter_map params({
		{"func", sys::make_param(this->_func)},
		{"waves", sys::make_param(waves_wrappper)},
		{"out_grid", sys::make_param(this->_outgrid, validate_grid<T,3>)},
	}, true);
	in >> params;
	validate_shape(this->_waves.shape(), "plain_wave.waves");
}

template <class T>
void
arma::generator::Plain_wave_model<T>::write(std::ostream& out) const {
	out << "func=" << this->_func
		<< ",waves=" << Array_wrapper<wave_type>(this->_waves);
}


template class arma::generator::Plain_wave_model<ARMA_REAL_TYPE>;
