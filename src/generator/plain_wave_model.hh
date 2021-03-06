#ifndef PLAIN_WAVE_HH
#define PLAIN_WAVE_HH

#include <istream>
#include <ostream>

#include "basic_model.hh"
#include "discrete_function.hh"
#include "physical_constants.hh"
#include "plain_wave_profile.hh"
#include "types.hh"

namespace arma {

	namespace generator {

		/**
		\brief Uses a simple sum of sines and cosines, small-amplitude waves.
		\ingroup generators

		Individual amplitudes, wave numbers, phases and velocities are set via
		configuration file.
		*/
		template<class T>
		class Plain_wave_model: public Basic_model<T> {

		public:
			typedef blitz::TinyVector<T,5> wave_type;
			typedef Array1D<wave_type> array_type;
			typedef Plain_wave_profile function_type;
			using typename Basic_model<T>::grid_type;

		private:
			/// Wave profile function.
			function_type _func = function_type::Cosine;
			array_type _waves;

		public:
			inline function_type
			get_function() const noexcept {
				return this->_func;
			}

			inline T
			get_shift() const noexcept {
				using namespace constants;
				return (this->_func == function_type::Cosine)
					? pi_div_2<T>
					: T(0);
			}

			inline const array_type&
			waves() const noexcept {
				return this->_waves;
			}

			inline int
			num_waves() const noexcept {
				return this->_waves.extent(0);
			}

			void validate() const override;

			Array3D<T>
			generate() override;

			inline T
			amplitude(int i) const noexcept {
				return this->_waves(i)(0);
			}

			inline T
			wavenum_x(int i) const noexcept {
				return this->_waves(i)(1);
			}

			inline T
			wavenum_y(int i) const noexcept {
				return this->_waves(i)(2);
			}

			inline T
			velocity(int i) const noexcept {
				return this->_waves(i)(3);
			}

			inline T
			phase(int i) const noexcept {
				return this->_waves(i)(4);
			}

		protected:
			void write(std::ostream& out) const override;
			void read(std::istream& in) override;

		private:
			template <class Func>
			Array3D<T>
			do_generate(Func elevation);

		};

	}

}

#endif // PLAIN_WAVE_HH
