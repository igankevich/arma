#include "plain_wave.hh"
#include "physical_constants.hh"
#include "validators.hh"
#include "params.hh"
#include "domain.hh"

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
arma::generator::Plain_wave_model<T>::generate(
	Discrete_function<T,3>& zeta,
	const Domain3D& subdomain
) {
	using constants::_2pi;
	const T shift = get_shift();
	const Shape3D& lbound = subdomain.lbound();
	const Shape3D& ubound = subdomain.ubound();
	const int t0 = lbound(0);
	const int j0 = lbound(1);
	const int k0 = lbound(2);
	const int t1 = ubound(0);
	const int j1 = ubound(1);
	const int k1 = ubound(2);
	const Domain<T,3> dom(
		zeta.grid().length() * (ubound-lbound) / zeta.grid().num_points(),
		ubound-lbound+1
	);
	#if ARMA_OPENMP
	#pragma omp parallel for collapse(3)
	#endif
	for (int t = t0; t <= t1; t++) {
		for (int j = j0; j <= j1; j++) {
			for (int k = k0; k <= k1; k++) {
				const T x = dom(j, 1);
				const T y = dom(k, 2);
				zeta(t, j, k) = blitz::sum(
					_amplitudes *
					blitz::sin(
						_2pi<T>*_wavenumbers*x - _velocities*t
						+ _phases + shift
					)
				);
			}
		}
	}
}


template <class T>
void
arma::generator::Plain_wave_model<T>::read(std::istream& in) {
	std::string func;
	Array_wrapper<T> wamplitudes(this->_amplitudes);
	Array_wrapper<T> wwavenumbers(this->_wavenumbers);
	Array_wrapper<T> wphases(this->_phases);
	Array_wrapper<T> wvelocitites(this->_velocities);
	sys::parameter_map params({
		{"func", sys::make_param(this->_func)},
		{"amplitudes", sys::make_param(wamplitudes)},
		{"wavenumbers", sys::make_param(wwavenumbers)},
		{"phases", sys::make_param(wphases)},
		{"velocities", sys::make_param(wvelocitites)},
	}, true);
	in >> params;
	validate_shape(this->_amplitudes.shape(), "plain_wave.amplitudes");
	validate_shape(this->_wavenumbers.shape(), "plain_wave.wavenumbers");
	validate_shape(this->_phases.shape(), "plain_wave.phases");
	validate_shape(this->_velocities.shape(), "plain_wave.velocities");
}

template <class T>
void
arma::generator::Plain_wave_model<T>::write(std::ostream& out) const {
	out << "func=" << this->_func
		<< ",amplitudes=" << Array_wrapper<T>(this->_amplitudes)
		<< ",wavenumbers=" << Array_wrapper<T>(this->_wavenumbers)
		<< ",phases=" << Array_wrapper<T>(this->_phases)
		<< ",velocities=" << Array_wrapper<T>(this->_velocities);
}


template class arma::generator::Plain_wave_model<ARMA_REAL_TYPE>;
