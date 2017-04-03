#include "plain_wave.hh"
#include "physical_constants.hh"
#include "validators.hh"
#include "params.hh"

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
arma::bits::operator>>(std::istream& in, Function& rhs) {
	std::string name;
	in >> std::ws >> name;
	if (name == "sin") {
		rhs = Function::Sine;
	} else if (name == "cos") {
		rhs = Function::Cosine;
	} else {
		in.setstate(std::ios::failbit);
		std::clog << "Invalid plain wave function: " << name << std::endl;
		throw std::runtime_error("bad function");
	}
	return in;
}

const char*
arma::bits::to_string(Function rhs) {
	switch (rhs) {
		case Function::Sine: return "sin";
		case Function::Cosine: return "cos";
		default: return "UNKNOWN";
	}
}


template <class T>
void
arma::Plain_wave<T>::generate(Array3D<T>& zeta, const Domain3D& subdomain) {
	using constants::_2pi;
	const T shift = get_shift();
	const Shape3D& lbound = subdomain.lbound();
	const Shape3D& ubound = subdomain.ubound();
	const int t0 = lbound(0);
	const int x0 = lbound(1);
	const int y0 = lbound(2);
	const int t1 = ubound(0);
	const int x1 = ubound(1);
	const int y1 = ubound(2);
	#if ARMA_OPENMP
	#pragma omp parallel for collapse(3)
	#endif
	for (int t = t0; t <= t1; t++) {
		for (int x = x0; x <= x1; x++) {
			for (int y = y0; y <= y1; y++) {
				zeta(t, x, y) = blitz::sum(
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
std::istream&
arma::operator>>(std::istream& in, Plain_wave<T>& rhs) {
	std::string func;
	Array_wrapper<T> wamplitudes(rhs._amplitudes);
	Array_wrapper<T> wwavenumbers(rhs._wavenumbers);
	Array_wrapper<T> wphases(rhs._phases);
	Array_wrapper<T> wvelocitites(rhs._velocities);
	sys::parameter_map params({
		{"func", sys::make_param(rhs._func)},
		{"amplitudes", sys::make_param(wamplitudes)},
		{"wavenumbers", sys::make_param(wwavenumbers)},
		{"phases", sys::make_param(wphases)},
		{"velocities", sys::make_param(wvelocitites)},
	}, true);
	in >> params;
	validate_shape(rhs._amplitudes.shape(), "plain_wave.amplitudes");
	validate_shape(rhs._wavenumbers.shape(), "plain_wave.wavenumbers");
	validate_shape(rhs._phases.shape(), "plain_wave.phases");
	validate_shape(rhs._velocities.shape(), "plain_wave.velocities");
	return in;
}

template <class T>
std::ostream&
arma::operator<<(std::ostream& out, const Plain_wave<T>& rhs) {
	return out
		<< "func=" << rhs._func
		<< ",amplitudes=" << Array_wrapper<T>(rhs._amplitudes)
		<< ",wavenumbers=" << Array_wrapper<T>(rhs._wavenumbers)
		<< ",phases=" << Array_wrapper<T>(rhs._phases)
		<< ",velocities=" << Array_wrapper<T>(rhs._velocities);
}


template class arma::Plain_wave<ARMA_REAL_TYPE>;
template std::ostream&
arma::operator<<(std::ostream& out, const Plain_wave<ARMA_REAL_TYPE>& rhs);
template std::istream&
arma::operator>>(std::istream& in, Plain_wave<ARMA_REAL_TYPE>& rhs);
