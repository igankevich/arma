#ifndef PLAIN_WAVE_HH
#define PLAIN_WAVE_HH

#include <cmath>
#include <string>
#include <iomanip>
#include <istream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include "params.hh"
#include "types.hh"
#include "validators.hh"

namespace arma {

	template<class T>
	class Array_wrapper {

		Array1D<T>& _arr;

	public:

		explicit
		Array_wrapper(Array1D<T>& arr):
		_arr(arr)
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
	};

	namespace bits {

		enum struct Function {
			Sine,
			Cosine
		};

		std::istream&
		operator>>(std::istream& in, Function& rhs) {
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
		to_string(Function rhs) {
			switch (rhs) {
				case Function::Sine: return "sin";
				case Function::Cosine: return "cos";
				default: return "UNKNOWN";
			}
		}

		std::ostream&
		operator<<(std::ostream& out, const Function& rhs) {
			return out << to_string(rhs);
		}

	}

	template<class T>
	class Plain_wave_model {

		bits::Function _func = bits::Function::Cosine;
		Array1D<T> _amplitudes;
		Array1D<T> _wavenumbers;
		Array1D<T> _phases;
		T _velocity = T(0.5);

	public:

		void
		operator()(Array3D<T>& zeta) {
			operator()(zeta, zeta.domain());
		}

		void
		operator()(Array3D<T>& zeta, const Domain3D& subdomain) {
			generate(zeta, subdomain);
		}

		friend std::istream&
		operator>>(std::istream& in, Plain_wave_model& rhs) {
			std::string func;
			Array_wrapper<T> wamplitudes(rhs._amplitudes);
			Array_wrapper<T> wwavenumbers(rhs._wavenumbers);
			Array_wrapper<T> wphases(rhs._phases);
			sys::parameter_map params({
			    {"func", sys::make_param(rhs._func)},
			    {"amplitudes", sys::make_param(wamplitudes)},
			    {"wavenumbers", sys::make_param(wwavenumbers)},
			    {"phases", sys::make_param(wphases)},
			    {"velocity", sys::make_param(rhs._velocity)},
			}, true);
			in >> params;
			validate_shape(rhs._amplitudes.shape(), "plain_wave.amplitudes");
			validate_shape(rhs._wavenumbers.shape(), "plain_wave.wavenumbers");
			validate_shape(rhs._phases.shape(), "plain_wave.phases");
			return in;
		}

		friend std::ostream&
		operator<<(std::ostream& out, const Plain_wave_model& rhs) {
			return out
				<< "func=" << rhs._func
				<< ",amplitudes=" << rhs._amplitudes
				<< ",wavenumbers=" << rhs._wavenumbers
				<< ",phases=" << rhs._phases
				<< ",velocity=" << rhs._velocity;
		}

	private:


		void
		generate(Array3D<T>& zeta, const Domain3D& subdomain) {
			T shift = 0;
			if (_func == bits::Function::Cosine) {
				shift = T(0.5) * T(M_PI);
			}
			const size3& lbound = subdomain.lbound();
			const size3& ubound = subdomain.ubound();
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
								_wavenumbers * (x - _velocity*t)
								+ _phases + shift
							)
						);
					}
				}
			}
		}

	};

}

#endif // PLAIN_WAVE_HH
