#ifndef PLAIN_WAVE_HH
#define PLAIN_WAVE_HH

#include <istream>
#include <ostream>
#include "types.hh"
#include "discrete_function.hh"

namespace arma {

	namespace bits {

		enum struct Function {
			Sine,
			Cosine
		};

		std::istream&
		operator>>(std::istream& in, Function& rhs);

		const char*
		to_string(Function rhs);

		inline std::ostream&
		operator<<(std::ostream& out, const Function& rhs) {
			return out << to_string(rhs);
		}

	}

	template<class T>
	class Plain_wave {

	public:
		typedef Array1D<T> array_type;
		typedef bits::Function function_type;

	private:
		function_type _func = function_type::Cosine;
		array_type _amplitudes;
		array_type _wavenumbers;
		array_type _phases;
		array_type _velocities;

	public:
		inline void
		operator()(Discrete_function<T,3>& zeta) {
			generate(zeta);
		}

		inline function_type
		get_function() const noexcept {
			return _func;
		}

		inline T
		get_shift() const noexcept {
			T shift = 0;
			if (_func == function_type::Cosine) {
				shift = T(0.5) * T(M_PI);
			}
			return shift;
		}

		inline const array_type&
		amplitudes() const noexcept {
			return _amplitudes;
		}

		inline const array_type&
		phases() const noexcept {
			return _phases;
		}

		inline const array_type&
		wavenumbers() const noexcept {
			return _wavenumbers;
		}

		inline const array_type&
		velocities() const noexcept {
			return _velocities;
		}

		inline int
		num_waves() const noexcept {
			return _amplitudes.size();
		}

		template <class X>
		friend std::istream&
		operator>>(std::istream& in, Plain_wave<X>& rhs);

		template <class X>
		friend std::ostream&
		operator<<(std::ostream& out, const Plain_wave<X>& rhs);

	private:
		void
		generate(Discrete_function<T,3>& zeta);

	};

	template <class T>
	std::istream&
	operator>>(std::istream& in, Plain_wave<T>& rhs);

	template <class T>
	std::ostream&
	operator<<(std::ostream& out, const Plain_wave<T>& rhs);

}

#endif // PLAIN_WAVE_HH
