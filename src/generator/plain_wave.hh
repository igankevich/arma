#ifndef PLAIN_WAVE_HH
#define PLAIN_WAVE_HH

#include <istream>
#include <ostream>
#include "types.hh"
#include "discrete_function.hh"
#include "basic_model.hh"

namespace arma {

	namespace generator {

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

		/**
		\brief Uses a simple sum of sines and cosines, small-amplitude waves.

		Individual amplitudes, wave numbers, phases and velocities are set via
		configuration file.
		*/
		template<class T>
		class Plain_wave_model: public Basic_model<T> {

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
				operator()(zeta, zeta.domain());
			}

			inline void
			operator()(Discrete_function<T,3>& zeta, const Domain3D& subdomain) {
				generate(zeta, subdomain);
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

			inline Array3D<T>
			generate() override {
				Discrete_function<T,3> zeta;
				zeta.resize(this->grid().num_points());
				zeta.setgrid(this->grid());
				operator()(zeta, zeta.domain());
				return zeta;
			}

			void determine_coefficients() override {}

			T white_noise_variance() const override {
				return T(0);
			}

			void validate() const override {}

			virtual void
			operator()(
				Array3D<T>& zeta,
				Array3D<T>& eps,
				const Domain3D& subdomain
			) override {
				throw std::runtime_error("not implemented");
			}

		protected:
			void write(std::ostream& out) const override;
			void read(std::istream& in) override;

		private:
			void
			generate(Discrete_function<T,3>& zeta, const Domain3D& subdomain);

		};

	}

}

#endif // PLAIN_WAVE_HH
